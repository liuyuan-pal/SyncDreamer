import os

import numpy as np
from argparse import ArgumentParser

import torch
from skimage.io import imread
from tqdm import tqdm
from torchmetrics.functional import structural_similarity_index_measure
import lpips


def compute_psnr_float(img_gt, img_pr):
    img_gt = img_gt.reshape([-1, 3]).astype(np.float32)
    img_pr = img_pr.reshape([-1, 3]).astype(np.float32)
    mse = np.mean((img_gt - img_pr) ** 2, 0)
    mse = np.mean(mse)
    psnr = 10 * np.log10(1 / mse)
    return psnr

def color_map_forward(rgb):
    dim = rgb.shape[-1]
    if dim==3:
        return rgb.astype(np.float32)/255
    else:
        rgb = rgb.astype(np.float32)/255
        rgb, alpha = rgb[:,:,:3], rgb[:,:,3:]
        rgb = rgb * alpha + (1-alpha)
        return rgb

def main():
    parser = ArgumentParser()
    parser.add_argument('--gt',type=str)
    parser.add_argument('--pr',type=str)
    parser.add_argument('--name',type=str)
    args = parser.parse_args()

    num_images = 16
    gt_dir = args.gt
    pr_dir = args.pr

    lpips_fn = lpips.LPIPS(net='vgg').cuda().eval()
    psnrs, ssims, lpipss = [], [], []
    for k in tqdm(range(num_images)):
        img_gt_int = imread(os.path.join(gt_dir, f'{k:03}.png'))
        img_pr_int = imread(os.path.join(pr_dir, f'{k:03}.png'))

        img_gt = color_map_forward(img_gt_int)
        img_pr = color_map_forward(img_pr_int)
        psnr = compute_psnr_float(img_gt, img_pr)

        with torch.no_grad():
            img_gt_tensor = torch.from_numpy(img_gt.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            img_pr_tensor = torch.from_numpy(img_pr.astype(np.float32)).permute(2,0,1).unsqueeze(0).cuda()
            ssim = float(structural_similarity_index_measure(img_pr_tensor, img_gt_tensor).flatten()[0].cpu().numpy())
            gt_img_th, pr_img_th = img_gt_tensor*2-1, img_pr_tensor*2-1
            score = float(lpips_fn(gt_img_th, pr_img_th).flatten()[0].cpu().numpy())

        ssims.append(ssim)
        lpipss.append(score)
        psnrs.append(psnr)


    msg=f'{args.name}\t{np.mean(psnrs):.5f}\t{np.mean(ssims):.5f}\t{np.mean(lpipss):.5f}'
    print(msg)
    with open('/cfs-cq-dcc/rondyliu/nvs.log','a') as f:
        f.write(msg+'\n')


if __name__=="__main__":
    main()