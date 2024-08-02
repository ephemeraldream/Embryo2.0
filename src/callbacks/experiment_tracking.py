import os
from typing import TYPE_CHECKING, Dict, Optional

from clearml import OutputModel, Task
from pytorch_lightning import Callback, LightningModule, Trainer

from src.config import ExperimentConfig
from src.logger import LOGGER

from lightning.pytorch.callbacks import ModelCheckpoint


class ClearMLTracking(Callback):
    def __init__(
            self,
            cfg: ExperimentConfig,
            label_enumeration: Optional[Dict[str, int]] = None,

    ):
        super().__init__()
        self.cfg = cfg
        self.label_enumeration = label_enumeration
        self.task: Optional[Task] = None
        self.output_model: Optional[OutputModel] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._setup_task()

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        checkpoint_path = select_checkpoint_for_export(trainer, on_valid=True)
        trainer.save_checkpoint(checkpoint_path)
        LOGGER.info('Saved Validation checkpoint: %s.', checkpoint_path)

    def _setup_task(self) -> None:
        Task.force_requirements_env_freeze()
        self.task = Task.init(
            project_name=self.cfg.project_name,
            task_name=self.cfg.experiment_name,
            output_uri=True,
        )
        self.task.connect_configuration(configuration=self.cfg.model_dump())
        self.output_model = OutputModel(task=self.task, label_enumeration=self.label_enumeration)


def select_checkpoint_for_export(trainer: Trainer, on_valid=False) -> str:
    checkpoint_cb: Optional[ModelCheckpoint] = trainer.checkpoint_callback
    if checkpoint_cb is not None:
        if on_valid:
            checkpoint_path = checkpoint_cb.last_model_path
        else:
            checkpoint_path = checkpoint_cb.best_model_path
        if os.path.isfile(checkpoint_path):
            LOGGER.info(f'Selected (VALID? {on_valid}) best checkpoint: %s', checkpoint_path)
            return checkpoint_path
        else:
            LOGGER.warning(
                "Couldn't find the best checkpoint, probably callback haven't been called yet. (func SELECT_CP)")

    checkpoint_path = os.path.join(trainer.log_dir, 'checkpoint-from-trainer.pth')
    trainer.save_checkpoint(checkpoint_path)
    LOGGER.info('Saved checkpoint: %s.', checkpoint_path)
    return checkpoint_path
