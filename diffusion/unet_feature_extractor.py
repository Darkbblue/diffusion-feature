import torch
from diffusers import UNet2DConditionModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from typing import Any, Dict, List, Optional, Tuple, Union

# following diffusers' implementation
# (https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py)
# (or: anaconda3/envs/xxx/lib/python3.9/site-packages/diffusers/models/unet_2d_condition.py)
class UnetFeatureExtractor(UNet2DConditionModel):
	# rewrite their forward function to output mid-layer values as feature
	# refer to that link if you want to change to output other values
	def extract_feature(
		self,
		sample: torch.FloatTensor,
		timestep: Union[torch.Tensor, float, int],
		encoder_hidden_states: torch.Tensor,
		class_labels: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		return_dict: bool = True,
	) -> Union[UNet2DConditionOutput, Tuple]:
		r"""
		Args:
			sample (`torch.FloatTensor`): (batch, channel, height, width) noisy inputs tensor
			timestep (`torch.FloatTensor` or `float` or `int`): (batch) timesteps
			encoder_hidden_states (`torch.FloatTensor`): (batch, sequence_length, feature_dim) encoder hidden states
			return_dict (`bool`, *optional*, defaults to `True`):
				Whether or not to return a [`models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain tuple.

		Returns:
			[`~models.unet_2d_condition.UNet2DConditionOutput`] or `tuple`:
			[`~models.unet_2d_condition.UNet2DConditionOutput`] if `return_dict` is True, otherwise a `tuple`. When
			returning a tuple, the first element is the sample tensor.
		"""
		# By default samples have to be AT least a multiple of the overall upsampling factor.
		# The overall upsampling factor is equal to 2 ** (# num of upsampling layears).
		# However, the upsampling interpolation output size can be forced to fit any upsampling size
		# on the fly if necessary.
		default_overall_up_factor = 2**self.num_upsamplers

		# upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
		forward_upsample_size = False
		upsample_size = None

		if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
			logger.info("Forward upsample size to force interpolation output size.")
			forward_upsample_size = True

		# prepare attention_mask
		if attention_mask is not None:
			attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
			attention_mask = attention_mask.unsqueeze(1)

		# 0. center input if necessary
		if self.config.center_input_sample:
			sample = 2 * sample - 1.0

		# 1. time
		timesteps = timestep
		if not torch.is_tensor(timesteps):
			# TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
			# This would be a good case for the `match` statement (Python 3.10+)
			is_mps = sample.device.type == "mps"
			if isinstance(timestep, float):
				dtype = torch.float32 if is_mps else torch.float64
			else:
				dtype = torch.int32 if is_mps else torch.int64
			timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
		elif len(timesteps.shape) == 0:
			timesteps = timesteps[None].to(sample.device)

		# broadcast to batch dimension in a way that's compatible with ONNX/Core ML
		timesteps = timesteps.expand(sample.shape[0])

		t_emb = self.time_proj(timesteps)

		# timesteps does not contain any weights and will always return f32 tensors
		# but time_embedding might actually be running in fp16. so we need to cast here.
		# there might be better ways to encapsulate this.
		t_emb = t_emb.to(dtype=self.dtype)
		emb = self.time_embedding(t_emb)

		if self.class_embedding is not None:
			if class_labels is None:
				raise ValueError("class_labels should be provided when num_class_embeds > 0")

			if self.config.class_embed_type == "timestep":
				class_labels = self.time_proj(class_labels)

			class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
			emb = emb + class_emb

		# 2. pre-process
		sample = self.conv_in(sample)

		# 3. down
		down_block_res_samples = (sample,)
		for downsample_block in self.down_blocks:
			if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
				sample, res_samples = downsample_block(
					hidden_states=sample,
					temb=emb,
					encoder_hidden_states=encoder_hidden_states,
					attention_mask=attention_mask,
				)
			else:
				sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

			down_block_res_samples += res_samples

		# 4. mid
		sample = self.mid_block(
			sample, emb, encoder_hidden_states=encoder_hidden_states, attention_mask=attention_mask
		)


		# output

		if not return_dict:
			return (sample,)

		return UNet2DConditionOutput(sample=sample)
