import os

import pytorch_lightning
from pytorch_lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

# TODO src.callbacks.debug

from src.config import ExperimentConfig
from src.constants import PROJECT_ROOT
from src.datamodule import EmbryoDataModule
from src.lightning_module import EmbryoLightningModule
from src.callbacks.experiment_tracking import ClearMLTracking

def train(cfg:ExperimentConfig) -> None:
    pytorch_lightning.seed_everything(0)
    datamodule = EmbryoDataModule(cfg=cfg.data_config)

    callbacks = [
        ClearMLTracking(cfg),
        LearningRateMonitor(logging_interval='step'),
        ModelCheckpoint(save_top_k=3, monitor='valid_f1', mode='max',every_n_epochs=1)
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



