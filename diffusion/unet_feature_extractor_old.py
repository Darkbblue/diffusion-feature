import torch
from diffusers import UNet2DConditionModel
from typing import Any, Dict, List, Optional, Tuple, Union

# following diffusers' implementation
# (https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/unet_2d_condition.py)
class UnetFeatureExtractor(UNet2DConditionModel):
	# rewrite their forward function to output mid-layer values as feature
	# refer to that link if you want to change to output other values
	def extract_feature(
		self,
		sample: torch.FloatTensor,
		timestep: Union[torch.Tensor, float, int],
		encoder_hidden_states: torch.Tensor,
		class_labels: Optional[torch.Tensor] = None,
		# timestep_cond: Optional[torch.Tensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		cross_attention_kwargs: Optional[Dict[str, Any]] = None,
		added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
		down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
		mid_block_additional_residual: Optional[torch.Tensor] = None,
		encoder_attention_mask: Optional[torch.Tensor] = None,
		return_dict: bool = True,
	):
		# By default samples have to be AT least a multiple of the overall upsampling factor.
		# The overall upsampling factor is equal to 2 ** (# num of upsampling layers).
		# However, the upsampling interpolation output size can be forced to fit any upsampling size
		# on the fly if necessary.
		default_overall_up_factor = 2**self.num_upsamplers

		# upsample size should be forwarded when sample is not a multiple of `default_overall_up_factor`
		forward_upsample_size = False
		upsample_size = None

		if any(s % default_overall_up_factor != 0 for s in sample.shape[-2:]):
			logger.info("Forward upsample size to force interpolation output size.")
			forward_upsample_size = True

		# ensure attention_mask is a bias, and give it a singleton query_tokens dimension
		# expects mask of shape:
		#   [batch, key_tokens]
		# adds singleton query_tokens dimension:
		#   [batch,                    1, key_tokens]
		# this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
		#   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
		#   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
		if attention_mask is not None:
			# assume that mask is expressed as:
			#   (1 = keep,      0 = discard)
			# convert mask into a bias that can be added to attention scores:
			#       (keep = +0,     discard = -10000.0)
			attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
			attention_mask = attention_mask.unsqueeze(1)

		# convert encoder_attention_mask to a bias the same way we do for attention_mask
		if encoder_attention_mask is not None:
			encoder_attention_mask = (1 - encoder_attention_mask.to(sample.dtype)) * -10000.0
			encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

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

		# `Timesteps` does not contain any weights and will always return f32 tensors
		# but time_embedding might actually be running in fp16. so we need to cast here.
		# there might be better ways to encapsulate this.
		t_emb = t_emb.to(dtype=sample.dtype)

		emb = self.time_embedding(t_emb)

		if self.class_embedding is not None:
			if class_labels is None:
				raise ValueError("class_labels should be provided when num_class_embeds > 0")

			if self.config.class_embed_type == "timestep":
				class_labels = self.time_proj(class_labels)

				# `Timesteps` does not contain any weights and will always return f32 tensors
				# there might be better ways to encapsulate this.
				class_labels = class_labels.to(dtype=sample.dtype)

			class_emb = self.class_embedding(class_labels).to(dtype=sample.dtype)

			if self.config.class_embeddings_concat:
				emb = torch.cat([emb, class_emb], dim=-1)
			else:
				emb = emb + class_emb

		if self.config.addition_embed_type == "text":
			aug_emb = self.add_embedding(encoder_hidden_states)
			emb = emb + aug_emb
		elif self.config.addition_embed_type == "text_image":
			# Kadinsky 2.1 - style
			if "image_embeds" not in added_cond_kwargs:
				raise ValueError(
					f"{self.__class__} has the config param `addition_embed_type` set to 'text_image' which requires the keyword argument `image_embeds` to be passed in `added_cond_kwargs`"
				)

			image_embs = added_cond_kwargs.get("image_embeds")
			text_embs = added_cond_kwargs.get("text_embeds", encoder_hidden_states)

			aug_emb = self.add_embedding(text_embs, image_embs)
			emb = emb + aug_emb

		if self.time_embed_act is not None:
			emb = self.time_embed_act(emb)

		if self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_proj":
			encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states)
		elif self.encoder_hid_proj is not None and self.config.encoder_hid_dim_type == "text_image_proj":
			# Kadinsky 2.1 - style
			if "image_embeds" not in added_cond_kwargs:
				raise ValueError(
					f"{self.__class__} has the config param `encoder_hid_dim_type` set to 'text_image_proj' which requires the keyword argument `image_embeds` to be passed in  `added_conditions`"
				)

			image_embeds = added_cond_kwargs.get("image_embeds")
			encoder_hidden_states = self.encoder_hid_proj(encoder_hidden_states, image_embeds)

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
					cross_attention_kwargs=cross_attention_kwargs,
					encoder_attention_mask=encoder_attention_mask,
				)
			else:
				sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

			down_block_res_samples += res_samples

		if down_block_additional_residuals is not None:
			new_down_block_res_samples = ()

			for down_block_res_sample, down_block_additional_residual in zip(
				down_block_res_samples, down_block_additional_residuals
			):
				down_block_res_sample = down_block_res_sample + down_block_additional_residual
				new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

			down_block_res_samples = new_down_block_res_samples

		# 4. mid
		if self.mid_block is not None:
			sample = self.mid_block(
				sample,
				emb,
				encoder_hidden_states=encoder_hidden_states,
				attention_mask=attention_mask,
				cross_attention_kwargs=cross_attention_kwargs,
				encoder_attention_mask=encoder_attention_mask,
			)

		if mid_block_additional_residual is not None:
			sample = sample + mid_block_additional_residual


		# output

		if not return_dict:
			return (sample,)

		return UNet2DConditionOutput(sample=sample)
