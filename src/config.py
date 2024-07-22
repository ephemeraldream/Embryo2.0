from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple, Union

import yaml
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field


class _BaseValidatedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')

class DataConfig(_BaseValidatedConfig):
    img_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32
    train_size: float = 0.8
    num_workers: int = 6
    pin_memory: bool = True
    num_samples: int = 11000

class SerializableObject(_BaseValidatedConfig):
    target_class: str
    kwargs: Dict[str, Any] = Field(default_factory=dict)


class ModuleConfig(_BaseValidatedConfig):
    model_name: str = 'efficientnet_b5'
    pretrained: bool = True
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    binarization_threshold: float = 0.5

    optimizer: SerializableObject = SerializableObject(
        target_class='torch.optim.AdamW',
        kwargs={'lr': 1e-3, 'weight_decay': 1e-5},
    )



class TrainerConfig(_BaseValidatedConfig):
    min_epochs: int = 5  # prevents early stopping
    max_epochs: int = 13

    check_val_every_n_epoch: int = 2

    log_every_n_steps: int = 50

    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: Optional[Literal['norm', 'value']] = None
    deterministic: bool = False
    fast_dev_run: bool = False
    default_root_dir: Optional[Path] = None
    detect_anomaly: bool = False



class ExperimentConfig(_BaseValidatedConfig):
    project_name: str = 'EmbryoVision'
    experiment_name: str = 'object_detection'
    track_in_clearml: bool = True
    trainer_config: TrainerConfig = Field(default=TrainerConfig())
    data_config: DataConfig = Field(default=DataConfig())
    module_config: ModuleConfig = Field(default=ModuleConfig())

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'ExperimentConfig':
        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        return cls(**cfg)

    def to_yaml(self, path: Union[str, Path]) -> None:
        with open(path, 'w') as out_file:
            yaml.safe_dump(self.model_dump(), out_file, default_flow_style=False, sort_keys=False)

