import pytorch_lightning as pl
import numpy as np
import torch
import PIL
import os
from skimage.io import imread
import webdataset as wds
import PIL.Image as Image
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from ldm.base_utils import read_pickle, pose_inverse
import torchvision.transforms as transforms
import torchvision
from einops import rearrange

from ldm.util import prepare_inputs


class SyncDreamerTrainData(Dataset):
    def __init__(self, target_dir, input_dir, uid_set_pkl, image_size=256):
        self.default_image_size = 256
        self.image_size = image_size
        self.target_dir = Path(target_dir)
        self.input_dir = Path(input_dir)

        self.uids = read_pickle(uid_set_pkl)
        print('============= length of dataset %d =============' % len(self.uids))

        image_transforms = []
        image_transforms.extend([transforms.ToTensor(), transforms.Lambda(lambda x: rearrange(x * 2. - 1., 'c h w -> h w c'))])
        self.image_transforms = torchvision.transforms.Compose(image_transforms)
        self.num_images = 16

    def __len__(self):
        return len(self.uids)

    def load_im(self, path):
        img = imread(path)
        img = img.astype(np.float32) / 255.0
        mask = img[:,:,3:]
        img[:,:,:3] = img[:,:,:3] * mask + 1 - mask # white background
        img = Image.fromarray(np.uint8(img[:, :, :3] * 255.))
        return img, mask

    def process_im(self, im):
        im = im.convert("RGB")
        im = im.resize((self.image_size, self.image_size), resample=PIL.Image.BICUBIC)
        return self.image_transforms(im)

    def load_index(self, filename, index):
        img, _ = self.load_im(os.path.join(filename, '%03d.png' % index))
        img = self.process_im(img)
        return img

    def get_data_for_index(self, index):
        target_dir = os.path.join(self.target_dir, self.uids[index])
        input_dir = os.path.join(self.input_dir, self.uids[index])

        views = np.arange(0, self.num_images)
        start_view_index = np.random.randint(0, self.num_images)
        views = (views + start_view_index) % self.num_images

        target_images = []
        for si, target_index in enumerate(views):
            img = self.load_index(target_dir, target_index)
            target_images.append(img)
        target_images = torch.stack(target_images, 0)
        input_img = self.load_index(input_dir, start_view_index)

        K, azimuths, elevations, distances, cam_poses = read_pickle(os.path.join(input_dir, f'meta.pkl'))
        input_elevation = torch.from_numpy(elevations[start_view_index:start_view_index+1].astype(np.float32))
        return {"target_image": target_images, "input_image": input_img, "input_elevation": input_elevation}

    def __getitem__(self, index):
        data = self.get_data_for_index(index)
        return data

class SyncDreamerEvalData(Dataset):
    def __init__(self, image_dir):
        self.image_size = 256
        self.image_dir = Path(image_dir)
        self.crop_size = 20

        self.fns = []
        for fn in Path(image_dir).iterdir():
            if fn.suffix=='.png':
                self.fns.append(fn)
        print('============= length of dataset %d =============' % len(self.fns))

    def __len__(self):
        return len(self.fns)

    def get_data_for_index(self, index):
        input_img_fn = self.fns[index]
        elevation = int(Path(input_img_fn).stem.split('-')[-1])
        return prepare_inputs(input_img_fn, elevation, 200)

    def __getitem__(self, index):
        return self.get_data_for_index(index)

class SyncDreamerDataset(pl.LightningDataModule):
    def __init__(self, target_dir, input_dir, validation_dir, batch_size, uid_set_pkl, image_size=256, num_workers=4, seed=0, **kwargs):
        super().__init__()
        self.target_dir = target_dir
        self.input_dir = input_dir
        self.validation_dir = validation_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uid_set_pkl = uid_set_pkl
        self.seed = seed
        self.additional_args = kwargs
        self.image_size = image_size

    def setup(self, stage):
        if stage in ['fit']:
            self.train_dataset = SyncDreamerTrainData(self.target_dir, self.input_dir, uid_set_pkl=self.uid_set_pkl, image_size=256)
            self.val_dataset = SyncDreamerEvalData(image_dir=self.validation_dir)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset, seed=self.seed)
        return wds.WebLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        loader = wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
        return loader

    def test_dataloader(self):
        return wds.WebLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
