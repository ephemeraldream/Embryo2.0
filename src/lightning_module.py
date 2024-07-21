from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as func
import timm
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import functional as F
from torchmetrics import MeanMetric
from torchmetrics import MetricCollection
from torch.nn.functional import one_hot


from src.config import ModuleConfig
from src.metrics import get_classification_metrics, get_regression_metrics
from src.schedulers import get_cosine_schedule_with_warmup
from src.preprocess.tools import Tools
from src.serialization import load_object



class EmbryoLightningModule(LightningModule):
    def __init__(self, classes: Tuple[str, ...], cfg: ModuleConfig):
        super().__init__()
        self.cfg = cfg
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        cls_metrics = get_classification_metrics(
            num_classes=len(classes))

        reg_metrics = get_regression_metrics().clone()

        self._val_cls_metrics = cls_metrics.clone(prefix='val_cls_')
        self._test_cls_metrics = cls_metrics.clone(prefix='test_cls_')
        self._val_reg_metrics = reg_metrics.clone(prefix='val_reg_')
        self._test_reg_metrics = reg_metrics.clone(prefix='test_reg_')




        self.model = timm.create_model(
            num_classes=len(classes),
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            **cfg.model_kwargs,

        )
        self.model.reset_classifier(0)

        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,100)
        )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 125)
        )
        self.hole_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        features = self.model(images)

        reg = self.reg_head(features).view(-1, 25, 4)
        cls = self.cls_head(features).view(-1, 25, 5)
        hole = self.hole_head(features).view(-1)

        return reg, cls, hole

    def training_step(self, batch: List[torch.Tensor]) -> Dict[str, Tensor, Tensor, Tensor]:
        images, targets = batch
        reg_pred, cls_pred, hole_pred = self(images)

        reg_loss = F.mse_loss(reg_pred, targets[0])
        cls_loss = F.cross_entropy(cls_pred, targets[1])
        hole_loss = F.binary_cross_entropy(hole_pred, targets[2].float())

        loss = reg_loss + cls_loss + hole_loss

        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'reg_pred': reg_pred, 'cls_pred': cls_pred, 'hole_pred': hole_pred}

    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        images, targets = batch
        reg_pred, cls_pred, hole_pred = self(images)

        reg_loss = F.mse_loss(reg_pred, targets[0])
        cls_loss = F.cross_entropy(cls_pred, targets[1])
        hole_loss = F.binary_cross_entropy(hole_pred, targets[2].float())
        loss = reg_loss + cls_loss + hole_loss
        self._valid_loss(loss)

        # TODO : Точно будут ошибки. Не забыть поменять размерность.
        _, max_index = torch.max(cls_pred,2)
        one_hot_preds = one_hot(max_index, num_classes=cls_pred.shape[2])

        self._val_cls_metrics(one_hot_preds, targets[1])
        self._val_reg_metrics(reg_pred, targets[0])
        return reg_pred, one_hot_preds, torch.where(hole_pred > 0.5, torch.tensor(1), torch.tensor(0))


    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        images, targets = batch
        reg_pred, cls_pred, hole_pred = self(images)

        reg_loss = F.mse_loss(reg_pred, targets[0])
        cls_loss = F.cross_entropy(cls_pred, targets[1])
        hole_loss = F.binary_cross_entropy(hole_pred, targets[2].float())
        loss = reg_loss + cls_loss + hole_loss
        self._valid_loss(loss)
        # TODO : Тоже размерность.
        _, max_index = torch.max(cls_pred,2)
        one_hot_preds = one_hot(max_index, num_classes=cls_pred.shape[2])

        self._test_cls_metrics(one_hot_preds, targets[1])
        self._test_reg_metrics(reg_pred, targets[0])
        return reg_pred, one_hot_preds, torch.where(hole_pred > 0.5, torch.tensor(1), torch.tensor(0))



    def on_train_epoch_end(self) -> None:
        self.log(
                'mean_train_loss',
                self._train_loss.compute(),
                on_step=False,
                prog_bar=True,
                on_epoch=True,
            )
        self._train_loss.reset()

    def on_validation_epoch_end(self) -> None:
        self.log(
            'mean_valid_loss',
            self._valid_loss.compute(),
            on_step=False,
            prog_bar=True,
            on_epoch=True,
        )
        self._valid_loss.reset()

        self.log_dict(self._val_cls_metrics.compute(), prog_bar=True, on_epoch=True)
        self.log_dict(self._val_reg_metrics.compute(), prog_bar=True, on_epoch=True)
        self._val_cls_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_dict(self._test_cls_metrics.compute(), prog_bar=True, on_epoch=True)
        self.log_dict(self._test_reg_metrics.compute(), prog_bar=True, on_epoch=True)
        self._test_cls_metrics.reset()
        self._test_reg_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = load_object(
            self.cfg.optimizer.target_class,
        )(self.model.parameters(), **self.cfg.optimizer.kwargs)

        # TODO: parametrize
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=200,
            num_training_steps=self.trainer.estimated_stepping_batches,
            num_cycles=1.4,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            },
        }
