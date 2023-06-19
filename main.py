import argparse
import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
import torch.nn.functional as F
import tqdm
from diffusion.datasets import get_target_dataset
from diffusion.models import get_sd_model, get_scheduler_config
from diffusion.utils import LOG_DIR, get_formatstr
import torchvision.transforms as torch_transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

INTERPOLATIONS = {
    'bilinear': InterpolationMode.BILINEAR,
    'bicubic': InterpolationMode.BICUBIC,
    'lanczos': InterpolationMode.LANCZOS,
}


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def get_transform(interpolation=InterpolationMode.BICUBIC, size=512):
    transform = torch_transforms.Compose([
        torch_transforms.Resize(size, interpolation=interpolation),
        torch_transforms.CenterCrop(size),
        _convert_image_to_rgb,
        torch_transforms.ToTensor(),
        torch_transforms.Normalize([0.5], [0.5])
    ])
    return transform


def center_crop_resize(img, interpolation=InterpolationMode.BILINEAR):
    transform = get_transform(interpolation=interpolation)
    return transform(img)


def extract_one_feature(unet, latent, text_embeds, scheduler, shared_noise, args, latent_size=64):
    all_noise = shared_noise
    if args.dtype == 'float16':
        all_noise = all_noise.half()
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.half()
    t = [args.t]
    dtype = args.dtype

    with torch.inference_mode():
        batch_ts = torch.tensor(t).expand(latent.shape[0])
        noise = all_noise
        noised_latent = latent * (scheduler.alphas_cumprod[batch_ts] ** 0.5).view(-1, 1, 1, 1).to(device) + \
                        noise * ((1 - scheduler.alphas_cumprod[batch_ts]) ** 0.5).view(-1, 1, 1, 1).to(device)
        t_input = batch_ts.to(device).half() if dtype == 'float16' else batch_ts.to(device)
        text_input = text_embeds.expand(latent.shape[0], -1, -1)
        features = unet.extract_feature(noised_latent, t_input, encoder_hidden_states=text_input).sample
        features = features.detach().cpu()
    return features


def main():
    parser = argparse.ArgumentParser()

    # run args
    parser.add_argument('--version', type=str, default='2-1', help='Stable Diffusion model version')
    parser.add_argument('--img_size', type=int, default=512, choices=(256, 512))
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--dtype', type=str, default='float16', choices=('float16', 'float32'),
                        help='Model data type to use')
    parser.add_argument('--interpolation', type=str, default='bicubic', help='Resize interpolation type')
    parser.add_argument('--n_workers', type=int, default=1, help='Number of workers to split the dataset across')
    parser.add_argument('--worker_idx', type=int, default=0, help='Index of worker to use')
    parser.add_argument('--load_stats', action='store_true', help='Load saved stats to compute acc')
    parser.add_argument('--t', type=int, help='Timesteps to compute features')

    parser.add_argument('--input_dir', type=str)
    parser.add_argument('--output_dir', type=str, default='./output/')

    args = parser.parse_args()

    # make run output folder
    os.makedirs(args.output_dir, exist_ok=True)
    print(f'Run folder: {args.output_dir}')

    # set up dataset
    interpolation = INTERPOLATIONS[args.interpolation]
    transform = get_transform(interpolation, args.img_size)
    latent_size = args.img_size // 8
    # use a build-in dataset
    # target_dataset = get_target_dataset(args.dataset, train=args.split == 'train', transform=transform)
    # or use raw image input
    preprocess = get_transform()
    # img = preprocess(Image.open('./cifar.jpg'))
    # target_dataset = [(img, None)]

    # load pretrained models
    vae, tokenizer, text_encoder, unet, scheduler = get_sd_model(args)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)
    unet = unet.to(device)
    torch.backends.cudnn.benchmark = True

    # prepare prompt input
    # refer to https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L276
    prompts = ['']
    text_input = tokenizer(prompts, padding="max_length",
                           max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    embeddings = []
    with torch.inference_mode():
        for i in range(0, len(text_input.input_ids), 100):
            text_embeddings = text_encoder(
                text_input.input_ids[i: i + 100].to(device),
            )[0]
            embeddings.append(text_embeddings)
    text_embeddings = torch.cat(embeddings, dim=0)
    assert len(text_embeddings) == len(prompts)

    # subset of dataset to evaluate
    shared_noise = torch.randn((1, 4, latent_size, latent_size), device=device)
    tasks = sorted(os.listdir(args.input_dir))[args.worker_idx::args.n_workers]

    # main loop
    target_dataset = []
    for filename in sorted([f for f in os.listdir(args.input_dir) if f.endswith('.jpg')]):
        img = preprocess(Image.open(args.input_dir + filename))
        target_dataset.append((img, filename))

    # idxs = list(range(len(target_dataset)))
    # idxs_to_eval = idxs[args.worker_idx::args.n_workers]

    formatstr = get_formatstr(len(target_dataset) - 1)
    i = 0
    results = []
    pbar = tqdm.tqdm(total=len(target_dataset))
    while i < len(target_dataset):
        sublist = target_dataset[i:i+args.batch_size]
        i += len(sublist)

        x0s = []
        with torch.no_grad():
            for image, _ in sublist:
                img_input = image.to(device).unsqueeze(0)
                if args.dtype == 'float16':
                    img_input = img_input.half()
                x0 = vae.encode(img_input).latent_dist.mean
                x0 *= 0.18215
                x0s.append(x0)
        x0 = torch.concatenate(x0s, axis=0)

        # the following code computes latent for a whole mini-batch
        # but it is slower than the code above on my machine
        # image = torch.stack([item[0] for item in sublist])
        # with torch.no_grad():
        #     img_input = image.to(device)
        #     if args.dtype == 'float16':
        #         img_input = img_input.half()
        #     x0 = vae.encode(img_input).latent_dist.mean
        #     x0 *= 0.18215

        features = extract_one_feature(unet, x0, text_embeddings, scheduler, shared_noise, args, latent_size)
        for j, (_, filename) in enumerate(sublist):
            np.save(args.output_dir+filename.replace('.jpg', ''), features[j])

        pbar.update(len(sublist))


if __name__ == '__main__':
    main()
