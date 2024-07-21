import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from dvc.repo import Repo
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler
import torchvision
from src.config import DataConfig
from src.constants import PROJECT_ROOT
from src.dataset import EmbryoDataset
#from src.dataset_splitter import read_df, split_and_save_datasets
from src.logger import LOGGER
#from src.transform import get_train_transforms, get_valid_transforms



DEFAULT_DATA_PATH = PROJECT_ROOT / 'dataset'


class EmbryoDataModule(LightningDataModule):
    def __init__(
            self,
            cfg: DataConfig
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        self.data_path = Path(os.getenv('DATA_PATH', DEFAULT_DATA_PATH))

        self.data_train: Optional[EmbryoDataset] = None
        self.data_val: Optional[EmbryoDataset] = None
        self.data_test: Optional[EmbryoDataset] = None

        self.initialized: bool = False

    def prepare_data(self) -> None:
        img_str = str(self.data_path) + "\\images"
        directory = os.fsencode(img_str)
        dir_of_images = dict()
        for file in os.listdir(directory):
            raw_filename = str(os.fsdecode(file))[:-11]
            filename_torch = img_str + str(os.fsencode(file))[2:-1]
            img = torchvision.io.read_image(filename_torch)
            dir_of_images[raw_filename] = img
        torch.save(dir_of_images, str(self.data_path) + "\\torched\\torch_imgs")


    def setup(self, stage: str) -> None:
        # TODO Can be type conflict.
        img_folder = str(self.data_path) +  '\\images'
        if stage == 'fit':




    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            sampler=RandomSampler(self.data_train, num_samples=self.cfg.num_samples),
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.cfg.batch_size,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            shuffle=False,
        )


