import pytorch_lightning as pl
from torch.utils.data import Dataset
import webdataset as wds
from torch.utils.data.distributed import DistributedSampler
class DummyDataset(pl.LightningDataModule):
    def __init__(self,seed):
        super().__init__()

    def setup(self, stage):
        if stage in ['fit']:
            self.train_dataset = DummyData(True)
            self.val_dataset = DummyData(False)
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return wds.WebLoader(self.train_dataset, batch_size=1, num_workers=0, shuffle=False)

    def val_dataloader(self):
        return wds.WebLoader(self.val_dataset, batch_size=1, num_workers=0, shuffle=False)

    def test_dataloader(self):
        return wds.WebLoader(DummyData(False))

class DummyData(Dataset):
    def __init__(self,is_train):
        self.is_train=is_train

    def __len__(self):
        if self.is_train:
            return 99999999
        else:
            return 1

    def __getitem__(self, index):
        return {}




