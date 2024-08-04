#Неведомая хуййня
import os

import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

# from src.callbacks.experiment_tracking import ClearMLTracking

# TODO src.callbacks.debug
from config import ExperimentConfig
from constants import PROJECT_ROOT
from datamodule import EmbryoDataModule
from lightning_module import EmbryoLightningModule
from datetime import datetime

model_checkpoint = ModelCheckpoint(
    dirpath= str(PROJECT_ROOT) + "/dataset/LG/version",
    filename="EFFB6V8regNO_PRETRAINED_circles",
    save_top_k=1,
    save_weights_only=False,
    mode='min',
    monitor='val_loss',
    verbose=True)


def train(cfg:ExperimentConfig) -> None:
    pytorch_lightning.seed_everything(0)
    datamodule = EmbryoDataModule(cfg=cfg.data_config)

    callbacks = [
        # ClearMLTracking(cfg),
        model_checkpoint,
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(save_top_k=3, monitor='val_reg_MSE', mode='max',every_n_epochs=1),
    ]
    model = EmbryoLightningModule(cfg=cfg.module_config)
    # TODO : Something is wrong with callbacks
    trainer = Trainer(
        **dict(cfg.trainer_config),
        callbacks=callbacks,
    )
    trainer.fit(datamodule=datamodule, model=model)
    trainer.test(datamodule=datamodule, model=model)


if __name__ == '__main__':
    cfg_path = os.getenv('TRAIN_CFG_PATH', PROJECT_ROOT / 'configs' / 'train.yaml')
    train(cfg=ExperimentConfig.from_yaml(cfg_path))



