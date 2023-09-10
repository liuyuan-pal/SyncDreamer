import argparse
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from skimage.io import imsave
from PIL import Image

from ldm.models.diffusion.sync_dreamer import SyncMultiviewDiffusion
from ldm.util import instantiate_from_config, add_margin


def load_model(cfg,ckpt,strict=True):
    config = OmegaConf.load(cfg)
    model = instantiate_from_config(config.model)
    print(f'loading model from {ckpt} ...')
    ckpt = torch.load(ckpt,map_location='cpu')
    model.load_state_dict(ckpt['state_dict'],strict=strict)
    model = model.cuda().eval()
    return model

def prepare_inputs(image_path, elevation_input, crop_size=-1, image_size=256):
    image_input = Image.open(image_path)

    if crop_size!=-1:
        alpha_np = np.asarray(image_input)[:, :, 3]
        coords = np.stack(np.nonzero(alpha_np), 1)[:, (1, 0)]
        min_x, min_y = np.min(coords, 0)
        max_x, max_y = np.max(coords, 0)
        ref_img_ = image_input.crop((min_x, min_y, max_x, max_y))
        h, w = ref_img_.height, ref_img_.width
        scale = crop_size / max(h, w)
        h_, w_ = int(scale * h), int(scale * w)
        ref_img_ = ref_img_.resize((w_, h_), resample=Image.BICUBIC)
        image_input = add_margin(ref_img_, size=image_size)
    else:
        image_input = add_margin(image_input, size=max(image_input.height, image_input.width))
        image_input = image_input.resize((image_size, image_size), resample=Image.BICUBIC)

    image_input = np.asarray(image_input)
    image_input = image_input.astype(np.float32) / 255.0
    ref_mask = image_input[:, :, 3:]
    image_input[:, :, :3] = image_input[:, :, :3] * ref_mask + 1 - ref_mask  # white background
    image_input = image_input[:, :, :3] * 2.0 - 1.0
    image_input = torch.from_numpy(image_input.astype(np.float32))
    elevation_input = torch.from_numpy(np.asarray([np.deg2rad(elevation_input)], np.float32))
    return {"input_image": image_input, "input_elevation": elevation_input}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg',type=str, default='configs/syncdreamer.yaml')
    parser.add_argument('--ckpt',type=str, default='ckpt/syncdreamer-step80k.ckpt')
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--elevation', type=float, required=True)

    parser.add_argument('--sample_num', type=int, default=4)
    parser.add_argument('--crop_size', type=int, default=-1)
    parser.add_argument('--cfg_scale', type=float, default=2.0)
    parser.add_argument('--batch_view_num', type=int, default=8)
    parser.add_argument('--seed', type=int, default=6033)
    flags = parser.parse_args()

    torch.random.manual_seed(flags.seed)
    np.random.seed(flags.seed)

    model = load_model(flags.cfg, flags.ckpt, strict=True)
    assert isinstance(model, SyncMultiviewDiffusion)
    Path(f'{flags.output}').mkdir(exist_ok=True, parents=True)

    # prepare data
    data = prepare_inputs(flags.input, flags.elevation, flags.crop_size)
    for k, v in data.items():
        data[k] = v.unsqueeze(0).cuda()
        data[k] = torch.repeat_interleave(data[k], flags.sample_num, dim=0)
    x_sample = model.sample(data, flags.cfg_scale, flags.batch_view_num)

    B, N, _, H, W = x_sample.shape
    x_sample = (torch.clamp(x_sample,max=1.0,min=-1.0) + 1) * 0.5
    x_sample = x_sample.permute(0,1,3,4,2).cpu().numpy() * 255
    x_sample = x_sample.astype(np.uint8)

    for bi in range(B):
        output_fn = Path(flags.output)/ f'{bi}.png'
        imsave(output_fn, np.concatenate([x_sample[bi,ni] for ni in range(N)], 1))

if __name__=="__main__":
    main()

