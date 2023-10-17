import argparse
import os
from pathlib import Path

import torch
import numpy as np
import transforms3d.euler
from skimage.io import imread
from tqdm import tqdm

from ldm.base_utils import project_points, mask_depth_to_pts, pose_inverse, pose_apply, output_points, read_pickle
import open3d as o3d
import mesh2sdf
import nvdiffrast.torch as dr

DEPTH_MAX, DEPTH_MIN = 2.4, 0.6
DEPTH_VALID_MAX, DEPTH_VALID_MIN = 2.37, 0.63
def read_depth_objaverse(depth_fn):
    depth = imread(depth_fn)
    depth = depth.astype(np.float32) / 65535 * (DEPTH_MAX-DEPTH_MIN) + DEPTH_MIN
    mask = (depth > DEPTH_VALID_MIN) & (depth < DEPTH_VALID_MAX)
    return depth, mask

K, _, _, _, POSES = read_pickle(f'meta_info/camera-16.pkl')
H, W, NUM_IMAGES = 256, 256, 16
CACHE_DIR = './eval_mesh_pts'

def rasterize_depth_map(mesh,pose,K,shape):
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.triangles, dtype=np.int32)
    pts, depth = project_points(vertices,pose,K)
    # normalize to projection
    h, w = shape
    pts[:,0]=(pts[:,0]*2-w)/w
    pts[:,1]=(pts[:,1]*2-h)/h
    near, far = 5e-1, 1e2
    z = (depth-near)/(far-near)
    z = z*2 - 1
    pts_clip = np.concatenate([pts,z[:,None]],1)

    pts_clip = torch.from_numpy(pts_clip.astype(np.float32)).cuda()
    indices = torch.from_numpy(faces.astype(np.int32)).cuda()
    pts_clip = torch.cat([pts_clip,torch.ones_like(pts_clip[...,0:1])],1).unsqueeze(0)
    ctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(ctx, pts_clip, indices, (h, w)) # [1,h,w,4]
    depth = (rast[0,:,:,2]+1)/2*(far-near)+near
    mask = rast[0,:,:,-1]!=0
    return depth.cpu().numpy(), mask.cpu().numpy().astype(np.bool)

def ds_and_save(cache_dir, name, pts, cache=False):
    cache_dir.mkdir(exist_ok=True, parents=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    downpcd = pcd.voxel_down_sample(voxel_size=0.01)
    if cache:
        o3d.io.write_point_cloud(str(cache_dir/(name + '.ply')), downpcd)
    return downpcd

def get_points_from_mesh(mesh, name, cache=False):
    obj_name = name
    cache_dir = Path(CACHE_DIR)
    fn = cache_dir/f'{obj_name}.ply'
    if cache and fn.exists():
        pcd = o3d.io.read_point_cloud(str(fn))
        return np.asarray(pcd.points)


    pts = []
    for index in range(NUM_IMAGES):
        pose = POSES[index]
        depth, mask = rasterize_depth_map(mesh, pose, K, (H, W))
        pts_ = mask_depth_to_pts(mask, depth, K)
        pose_inv = pose_inverse(pose)
        pts.append(pose_apply(pose_inv, pts_))

    pts = np.concatenate(pts, 0).astype(np.float32)
    downpcd = ds_and_save(cache_dir, obj_name, pts, cache)
    return np.asarray(downpcd.points,np.float32)

def get_points_from_depth(depth_dir, obj_name):
    cache_dir = Path(CACHE_DIR)
    fn = cache_dir/f'{obj_name}.ply'
    if fn.exists():
        pcd = o3d.io.read_point_cloud(str(fn))
        return np.asarray(pcd.points)

    pts = []
    for k in range(NUM_IMAGES):
        depth, mask = read_depth_objaverse(os.path.join(depth_dir,f'{k:03}-depth.png'))
        pts_ = mask_depth_to_pts(mask, depth, K)
        pose_inv = pose_inverse(POSES[k])
        pts.append(pose_apply(pose_inv, pts_))

    pts = np.concatenate(pts, 0).astype(np.float32)
    downpcd = ds_and_save(cache_dir, obj_name, pts, True)
    return np.asarray(downpcd.points,np.float32)

def nearest_dist(pts0, pts1, batch_size=512):
    pts0 = torch.from_numpy(pts0.astype(np.float32)).cuda()
    pts1 = torch.from_numpy(pts1.astype(np.float32)).cuda()
    pn0, pn1 = pts0.shape[0], pts1.shape[0]
    dists = []
    for i in tqdm(range(0, pn0, batch_size), desc='evaluating...'):
        dist = torch.norm(pts0[i:i+batch_size,None,:] - pts1[None,:,:], dim=-1)
        dists.append(torch.min(dist,1)[0])
    dists = torch.cat(dists,0)
    return dists.cpu().numpy()

def norm_coords(vertices):
    max_pt = np.max(vertices, 0)
    min_pt = np.min(vertices, 0)
    scale = 1 / np.max(max_pt - min_pt)
    vertices = vertices * scale

    max_pt = np.max(vertices, 0)
    min_pt = np.min(vertices, 0)
    center = (max_pt + min_pt) / 2
    vertices = vertices - center[None, :]
    return vertices

def transform_gt(vertices, rot_angle):
    vertices = norm_coords(vertices)
    R = transforms3d.euler.euler2mat(-np.deg2rad(rot_angle), 0, 0, 'szyx')
    vertices = vertices @ R.T

    return vertices


def get_chamfer_iou(mesh_pr, mesh_gt, pr_name, gt_name, gt_dir, output):
    pts_pr = get_points_from_mesh(mesh_pr, pr_name)
    pts_gt = get_points_from_depth(gt_dir, gt_name)

    if output:
        output_points(f'vis_val/{pr_name}-mesh-pr.txt', mesh_pr.vertices)
        output_points(f'vis_val/{pr_name}-mesh-gt.txt', mesh_gt.vertices)
        output_points(f'vis_val/{pr_name}-pts-pr.txt', pts_pr)
        output_points(f'vis_val/{pr_name}-pts-gt.txt', pts_gt)

    # compute iou
    size = 64
    sdf_pr = mesh2sdf.compute(mesh_pr.vertices, mesh_pr.triangles, size, fix=False, return_mesh=False)
    sdf_gt = mesh2sdf.compute(mesh_gt.vertices, mesh_gt.triangles, size, fix=False, return_mesh=False)
    vol_pr = sdf_pr<0
    vol_gt = sdf_gt<0
    iou = np.sum(vol_pr & vol_gt)/np.sum(vol_gt | vol_pr)

    dist0 = nearest_dist(pts_pr, pts_gt, batch_size=4096)
    dist1 = nearest_dist(pts_gt, pts_pr, batch_size=4096)

    chamfer = (np.mean(dist0) + np.mean(dist1)) / 2
    return chamfer, iou

def get_gt_rotate_angle(object_name):
    angle = 0
    if object_name == 'sofa':
        angle -= np.pi / 2
    elif object_name in ['blocks', 'alarm', 'backpack', 'chicken', 'soap', 'grandfather', 'grandmother', 'lion', 'lunch_bag', 'mario', 'oil']:
        angle += np.pi / 2 * 3
    elif object_name in ['elephant', 'school_bus1']:
        angle += np.pi
    elif object_name in ['school_bus2', 'shoe', 'train', 'turtle']:
        angle += np.pi / 8 * 10
    elif object_name in ['sorter']:
        angle += np.pi / 8 * 5
    angle = np.rad2deg(angle)
    return angle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pr_mesh', type=str, required=True)
    parser.add_argument('--pr_name', type=str, required=True)
    parser.add_argument('--gt_dir', type=str, required=True)
    parser.add_argument('--gt_mesh', type=str, required=True)
    parser.add_argument('--gt_name', type=str, required=True)
    parser.add_argument('--output', action='store_true', default=False, dest='output')
    args = parser.parse_args()

    mesh_gt = o3d.io.read_triangle_mesh(args.gt_mesh)
    vertices_gt = np.asarray(mesh_gt.vertices)
    vertices_gt = transform_gt(vertices_gt, get_gt_rotate_angle(args.gt_name))
    mesh_gt.vertices = o3d.utility.Vector3dVector(vertices_gt)

    mesh_pr = o3d.io.read_triangle_mesh(args.pr_mesh)
    vertices_pr = np.asarray(mesh_pr.vertices)
    mesh_pr.vertices = o3d.utility.Vector3dVector(vertices_pr)
    chamfer, iou = get_chamfer_iou(mesh_pr, mesh_gt, args.pr_name, args.gt_name, args.gt_dir, args.output)

    results = f'{args.pr_name}\t{chamfer:.5f}\t{iou:.5f}'
    print(results)
    with open('/cfs-cq-dcc/rondyliu/geometry.log','a') as f:
        f.write(results+'\n')

if __name__=="__main__":
    main()