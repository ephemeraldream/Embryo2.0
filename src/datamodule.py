import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from dvc.repo import Repo
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split
import torchvision
from src.config import DataConfig
from src.constants import PROJECT_ROOT
from src.dataset import EmbryoDataset
# from src.dataset_splitter import read_df, split_and_save_datasets
from src.logger import LOGGER

# from src.transform import get_train_transforms, get_valid_transforms


DEFAULT_DATA_PATH = PROJECT_ROOT / 'dataset'


# TODO : Добавить dvc, всё логгировать.
class EmbryoDataModule(LightningDataModule):
    def __init__(
            self,
            cfg: DataConfig
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(logger=False)
        self.data_path = Path(os.getenv('DATA_PATH', DEFAULT_DATA_PATH))
        self.classes = ['empty', 'embryo', 'embryogas', 'onlygas', 'None']

        self.data_train: Optional[EmbryoDataset] = None
        self.data_val: Optional[EmbryoDataset] = None
        self.data_test: Optional[EmbryoDataset] = None

        self.initialized: bool = False

    def prepare_data(self) -> None:
        # TODO : Мб потом сместить сбор сырых данных сюда.
        pass

    @property
    def class_to_idx(self) -> Dict[str, int]:
        return {cls_name: idx for idx, cls_name in enumerate(self.classes)}

    @property
    def idx_to_class(self) -> Dict[int, str]:
        return {idx: cls_name for cls_name, idx in self.class_to_idx.items()}

    def setup(self, stage: str) -> None:
        # TODO Can be type conflict.
        img_folder = str(self.data_path) + '\\images'
        labels = torch.load(str(DEFAULT_DATA_PATH) + "\\torched\\torched_labels")
        images = torch.load(str(DEFAULT_DATA_PATH) + "\\torched\\torched_images")

        dataset = EmbryoDataset(images, labels)
        shuffled_dataset = Subset(dataset, torch.randperm(len(dataset)).tolist())
        train_length = int(0.8 * len(dataset))
        train, test = random_split(
            shuffled_dataset,
            [train_length, len(dataset) - train_length]
        )

        if stage == 'fit':
            shorter_length = int(0.8 * train_length)
            val_length = train_length - shorter_length

            self.data_train, self.data_val = random_split(
                train,
                [shorter_length, val_length]
            )


        elif stage == 'test':
            self.data_test = test

        self.initialized = True

    # TODO : Чекнуть все конфиги.
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
