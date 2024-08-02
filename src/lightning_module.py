from typing import Any, Dict, List, Tuple

import timm
import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.functional import one_hot
from torchmetrics import MeanMetric
from datetime import datetime
from pytorch_lightning.callbacks import ModelCheckpoint
from constants import PROJECT_ROOT

from config import ModuleConfig
from metrics import get_classification_metrics, get_regression_metrics
from schedulers import get_cosine_schedule_with_warmup
from serialization import load_object


class EmbryoLightningModule(LightningModule):

    def __init__(self, cfg: ModuleConfig):
        super().__init__()
        self.cfg = cfg
        self._train_loss = MeanMetric()
        self._valid_loss = MeanMetric()

        cls_metrics = get_classification_metrics(
            task = 'multiclass',
            num_classes=cfg.num_classes)

        reg_metrics = get_regression_metrics().clone()

        torch.cuda.empty_cache()

        self._val_cls_metrics = cls_metrics.clone(prefix='val_cls_')
        self._test_cls_metrics = cls_metrics.clone(prefix='test_cls_')
        self._val_reg_metrics = reg_metrics.clone(prefix='val_reg_')
        self._test_reg_metrics = reg_metrics.clone(prefix='test_reg_')




        self.model = timm.create_model(
            num_classes=cfg.num_classes,
            model_name=cfg.model_name,
            pretrained=cfg.pretrained,
            **cfg.model_kwargs,

        )
        # TODO : Изменение входных каналов на 1.
        self.modify_first_conv_layer()


        self.model.reset_classifier(0)

        self.reg_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512,100),
        )
        self.cls_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 125),
        )
        self.hole_head = torch.nn.Sequential(
            torch.nn.Linear(self.model.num_features, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid(),
        )

        self.save_hyperparameters()

    def forward(self, images: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        images = torch.squeeze(images, dim=2)
        # assert images.shape == (32,1,224,224)
        features = self.model(images)


        reg = self.reg_head(features).view(-1, 25, 4)
        cls = self.cls_head(features).view(-1, 25, 5)
        hole = self.hole_head(features).view(-1)

        return reg, cls, hole

    def training_step(self, batch: List[torch.Tensor]) -> Dict[str, Any]:
        images, targets = batch
        images = torch.squeeze(images, dim=2)
        #assert images.shape == (32,1,224,224)

        reg_pred, cls_pred, hole_pred = self(images)

        reg_loss = F.mse_loss(reg_pred, targets[0])
        cls_loss = F.cross_entropy(cls_pred, targets[1])
        hole_loss = F.binary_cross_entropy(hole_pred, targets[2].float())

        loss = reg_loss + cls_loss + hole_loss

        _, max_index = torch.max(cls_pred,2)
        one_hot_preds = one_hot(max_index, num_classes=cls_pred.shape[2])
        self._train_loss(loss)
        self.log('step_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'reg_pred': reg_pred, 'cls_pred': cls_pred, 'hole_pred': hole_pred}


    def validation_step(self, batch: List[torch.Tensor], batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        images, targets = batch
        images = torch.squeeze(images, dim=2)
        reg_pred, cls_pred, hole_pred = self(images)

        reg_loss = F.mse_loss(reg_pred, targets[0])
        cls_loss = F.cross_entropy(cls_pred, targets[1])
        hole_loss = F.binary_cross_entropy(hole_pred, targets[2].float())
        loss = reg_loss + cls_loss + hole_loss
        self._valid_loss(loss)

        # TODO : Точно будут ошибки. Не забыть поменять размерность.
        _, max_index = torch.max(cls_pred,2)
        one_hot_preds = one_hot(max_index, num_classes=cls_pred.shape[2])
        self.log('val_loss', self._valid_loss, on_epoch=True, prog_bar=True)

        self._val_cls_metrics(one_hot_preds, targets[1])
        self._val_reg_metrics(reg_pred, targets[0])
        return reg_pred, one_hot_preds, torch.where(hole_pred > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))



    def test_step(self, batch: List[torch.Tensor], batch_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        images, targets = batch
        images = torch.squeeze(images, dim=2)
        reg_pred, cls_pred, hole_pred = self(images)

        _, max_index = torch.max(cls_pred,2)
        one_hot_preds = one_hot(max_index, num_classes=cls_pred.shape[2])
        self._test_cls_metrics(one_hot_preds, targets[1])
        self._test_reg_metrics(reg_pred, targets[0])
        return reg_pred, one_hot_preds, torch.where(hole_pred > 0.5, torch.tensor(1).to(device), torch.tensor(0).to(device))



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
        self._test_reg_metrics.reset()
        self._test_cls_metrics.reset()

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

    def modify_first_conv_layer(self) -> None:
        original_conv = self.model.conv_stem
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=(3,3),
            stride=(2,2),
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        with torch.no_grad():
            new_conv.weight = nn.Parameter(torch.mean(original_conv.weight, dim=1, keepdim=True))
            if original_conv.bias is not None:
                new_conv.bias = original_conv.bias

        self.model.conv_stem = new_conv
