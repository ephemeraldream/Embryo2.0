import os
from pathlib import Path
import gdown
from typing import Dict, Optional
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, RandomSampler, Subset, random_split

from config import DataConfig
from constants import PROJECT_ROOT
from dataset import EmbryoDataset

# from src.dataset_splitter import read_df, split_and_save_datasets

# from src.transform import get_train_transforms, get_valid_transforms


DEFAULT_DATA_PATH = PROJECT_ROOT / 'dataset'


# TODO : Добавить dvc, всё логгировать.
class EmbryoDataModule(LightningDataModule):
    def __init__(
            self,
            cfg: DataConfig,
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
        dataset_path = os.path.join(PROJECT_ROOT, "dataset", "torched")
        backup_path = os.path.join(dataset_path, "backup")
        labels_path = os.path.join(dataset_path, "torched_labels_circles")  # Замените на правильный формат
        images_path = os.path.join(dataset_path, "images")  # Замените на правильный формат
        #
        # # Проверяем существование папок и создаем их при необходимости
        # os.makedirs(dataset_path, exist_ok=True)
        # os.makedirs(backup_path, exist_ok=True)
        #
        # # Функция для загрузки файла, если он не существует
        # labels_path = os.path.join(dataset_path, "labels.pt")
        # images_path = os.path.join(dataset_path, "images.pt")
        #
        # # Функция для скачивания файла, если он не существует
        # def download_if_not_exists(url, path):
        #     if not os.path.exists(path):
        #         os.makedirs(dataset_path, exist_ok=True)
        #         gdown.download(url, path, quiet=False)
        #
        # # Проверяем и скачиваем файлы при необходимости
        # download_if_not_exists('https://drive.google.com/uc?id=1HQXqQND5NuP97rkuZujM8FwY4kEIxAEB', labels_path)
        # download_if_not_exists('https://drive.google.com/uc?id=10jGdYGwqowfirF9nI0XTKM4TmKPqxotc', images_path)
        #
        # # Загружаем данные с помощью torch.load
        labels = torch.load(labels_path)
        images = torch.load(images_path)
        # Загружаем данные в переменные

        dataset = EmbryoDataset(images, labels)
        shuffled_dataset = Subset(dataset, torch.randperm(len(dataset)).tolist())
        train_length = int(0.8 * len(dataset))
        train, test = random_split(
            shuffled_dataset,
            [train_length, len(dataset) - train_length],
        )

        if stage == 'fit':
            shorter_length = int(0.8 * train_length)
            val_length = train_length - shorter_length

            self.data_train, self.data_val = random_split(
                train,
                [shorter_length, val_length],
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
