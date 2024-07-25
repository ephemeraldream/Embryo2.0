import os
from typing import TYPE_CHECKING, Dict, Optional

from clearml import OutputModel, Task
from pytorch_lightning import Callback, LightningModule, Trainer

from src.config import ExperimentConfig
from src.logger import LOGGER

if TYPE_CHECKING:
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
        self.output_model:Optional[OutputModel] = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self._setup_task()

    def on_test_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        final_checkpoint = select_checkpoint_for_export(trainer)
        LOGGER.info('Uploading checkpoint "%s" to ClearML', final_checkpoint)
        if self.output_model is None:
            msg = 'Expected output model to be initialized.'
            raise ValueError(msg)
        self.output_model.update_weights(weights_filename=final_checkpoint, auto_delete_file=True)

    def on_validation_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        checkpoint: Optional[ModelCheckpoint] = trainer.checkpoint_callback
        if checkpoint is None:
            LOGGER.info("THERE IS NO CHECKPOINT YET.")
        else:
            cp_path = checkpoint.last_model_path
            checkpoint_path = os.path.join(trainer.log_dir, 'checkpoint-from-trainer.pth')
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






def select_checkpoint_for_export(trainer: Trainer, on_test=True) -> str:
    checkpoint_cb: Optional[ModelCheckpoint] = trainer.checkpoint_callback
    if on_test:
        if checkpoint_cb is not None:
            checkpoint_path = checkpoint_cb.best_model_path
            if os.path.isfile(checkpoint_path):
                LOGGER.info('Selected best checkpoint: %s', checkpoint_path)
                return checkpoint_path
            else:
                LOGGER.warning("Couldn't find the best checkpoint, probably callback haven't been called yet.")
            checkpoint_path = os.path.join(trainer.log_dir, 'checkpoint-from-trainer.pth')
            trainer.save_checkpoint(checkpoint_path)
            LOGGER.info('Saved checkpoint: %s.', checkpoint_path)
            return checkpoint_path
    else:
        if checkpoint_cb is not None:
            checkpoint_path = checkpoint_cb.last_model_path
            if os.path.isfile(checkpoint_path):
                LOGGER.info('Selected of last valid checkpoint: %s', checkpoint_path)
                return checkpoint_path
            else:
                LOGGER.warning("Couldn't find the best checkpoint, probably callback haven't been called yet.")


        checkpoint_path = os.path.join(trainer.log_dir, 'checkpoint-from-trainer.pth')
        trainer.save_checkpoint(checkpoint_path)
        LOGGER.info('Saved checkpoint: %s.', checkpoint_path)
        return checkpoint_path


