import argparse

import imageio
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

import trimesh
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from skimage.io import imsave
from tqdm import tqdm

import mcubes

from ldm.base_utils import read_pickle, output_points
from renderer.renderer import NeuSRenderer, DEFAULT_SIDE_LENGTH
from ldm.util import instantiate_from_config

class ResumeCallBacks(Callback):
    def __init__(self):
        pass

    def on_train_start(self, trainer, pl_module):
        pl_module.optimizers().param_groups = pl_module.optimizers()._optimizer.param_groups

def render_images(model, output,):
    # render from model
    n = 180
    azimuths = (np.arange(n) / n * np.pi * 2).astype(np.float32)
    elevations = np.deg2rad(np.asarray([30] * n).astype(np.float32))
    K, _, _, _, poses = read_pickle(f'meta_info/camera-16.pkl')
    output_points
    h, w = 256, 256
    default_size = 256
    K = np.diag([w/default_size,h/default_size,1.0]) @ K
    imgs = []
    for ni in tqdm(range(n)):
        # R = euler2mat(azimuths[ni], elevations[ni], 0, 'szyx')
        # R = np.asarray([[0,-1,0],[0,0,-1],[1,0,0]]) @ R
        e, a = elevations[ni], azimuths[ni]
        row1 = np.asarray([np.sin(e)*np.cos(a),np.sin(e)*np.sin(a),-np.cos(e)])
        row0 = np.asarray([-np.sin(a),np.cos(a), 0])
        row2 = np.cross(row0, row1)
        R = np.stack([row0,row1,row2],0)
        t = np.asarray([0,0,1.5])
        pose = np.concatenate([R,t[:,None]],1)
        pose_ = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0)
        K_ = torch.from_numpy(K.astype(np.float32)).unsqueeze(0) # [1,3,3]

        coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1)[:, :, (1, 0)]  # h,w,2
        coords = coords.float()[None, :, :, :].repeat(1, 1, 1, 1)  # imn,h,w,2
        coords = coords.reshape(1, h * w, 2)
        coords = torch.cat([coords, torch.ones(1, h * w, 1, dtype=torch.float32)], 2)  # imn,h*w,3

        # imn,h*w,3 @ imn,3,3 => imn,h*w,3
        rays_d = coords @ torch.inverse(K_).permute(0, 2, 1)
        R, t = pose_[:, :, :3], pose_[:, :, 3:]
        rays_d = rays_d @ R
        rays_d = F.normalize(rays_d, dim=-1)
        rays_o = -R.permute(0, 2, 1) @ t  # imn,3,3 @ imn,3,1
        rays_o = rays_o.permute(0, 2, 1).repeat(1, h * w, 1)  # imn,h*w,3

        ray_batch = {
            'rays_o': rays_o.reshape(-1,3).cuda(),
            'rays_d': rays_d.reshape(-1,3).cuda(),
        }
        with torch.no_grad():
            image = model.renderer.render(ray_batch,False,5000)['rgb'].reshape(h,w,3)
        image = (image.cpu().numpy() * 255).astype(np.uint8)
        imgs.append(image)

    imageio.mimsave(f'{output}/rendering.mp4', imgs, fps=30)

def extract_fields(bound_min, bound_max, resolution, query_func, batch_size=64, outside_val=1.0):
    N = batch_size
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).cuda()
                    val = query_func(pts).detach()
                    outside_mask = torch.norm(pts,dim=-1)>=1.0
                    val[outside_mask]=outside_val
                    val = val.reshape(len(xs), len(ys), len(zs)).cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u

def extract_geometry(bound_min, bound_max, resolution, threshold, query_func, color_func, outside_val=1.0):
    u = extract_fields(bound_min, bound_max, resolution, query_func, outside_val=outside_val)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    vertex_colors = color_func(vertices)
    return vertices, triangles, vertex_colors

def extract_mesh(model, output, resolution=512):
    if not isinstance(model.renderer, NeuSRenderer): return
    bbox_min = -torch.ones(3)*DEFAULT_SIDE_LENGTH
    bbox_max = torch.ones(3)*DEFAULT_SIDE_LENGTH
    with torch.no_grad():
        vertices, triangles, vertex_colors = extract_geometry(bbox_min, bbox_max, resolution, 0, lambda x: model.renderer.sdf_network.sdf(x), lambda x: model.renderer.get_vertex_colors(x))

    # output geometry
    mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertex_colors)
    mesh.export(str(f'{output}/mesh.ply'))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_path', type=str, required=True)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-b', '--base', type=str, default='configs/neus.yaml')
    parser.add_argument('-l', '--log', type=str, default='output/renderer')
    parser.add_argument('-s', '--seed', type=int, default=6033)
    parser.add_argument('-g', '--gpus', type=str, default='0,')
    parser.add_argument('-r', '--resume', action='store_true', default=False, dest='resume')
    parser.add_argument('--fp16', action='store_true', default=False, dest='fp16')
    opt = parser.parse_args()
    # seed_everything(opt.seed)

    # configs
    cfg = OmegaConf.load(opt.base)
    name = opt.name
    log_dir, ckpt_dir = Path(opt.log) / name, Path(opt.log) / name / 'ckpt'
    cfg.model.params['image_path'] = opt.image_path
    cfg.model.params['log_dir'] = log_dir

    # setup
    log_dir.mkdir(exist_ok=True, parents=True)
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    trainer_config = cfg.trainer
    callback_config = cfg.callbacks
    model_config = cfg.model
    data_config = cfg.data

    data_config.params.seed = opt.seed
    data = instantiate_from_config(data_config)
    data.prepare_data()
    data.setup('fit')

    model = instantiate_from_config(model_config,)
    model.cpu()
    model.learning_rate = model_config.base_lr

    # logger
    logger = TensorBoardLogger(save_dir=log_dir, name='tensorboard_logs')
    callbacks=[]
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    callbacks.append(ModelCheckpoint(dirpath=ckpt_dir, filename="{epoch:06}", verbose=True, save_last=True, every_n_train_steps=callback_config.save_interval))

    # trainer
    trainer_config.update({
        "accelerator": "cuda", "check_val_every_n_epoch": None,
        "benchmark": True, "num_sanity_val_steps": 0,
        "devices": 1, "gpus": opt.gpus,
    })
    if opt.fp16:
        trainer_config['precision']=16

    if opt.resume:
        callbacks.append(ResumeCallBacks())
        trainer_config['resume_from_checkpoint'] = str(ckpt_dir / 'last.ckpt')
    else:
        if (ckpt_dir / 'last.ckpt').exists():
            raise RuntimeError(f"checkpoint {ckpt_dir / 'last.ckpt'} existing ...")
    trainer = Trainer.from_argparse_args(args=argparse.Namespace(), **trainer_config, logger=logger, callbacks=callbacks)

    trainer.fit(model, data)

    model = model.cuda().eval()

    render_images(model, log_dir)
    extract_mesh(model, log_dir)

if __name__=="__main__":
    main()