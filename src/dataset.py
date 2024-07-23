from pathlib import Path
from typing import Optional, Tuple, Dict

import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import ToPILImage
import jpeg4py as jpeg
import numpy as np
import pandas as pd
from torch import Tensor
from torch.utils.data import Dataset
from src.transform import get_basic_transform


class EmbryoDataset(Dataset):
    def __init__(
            self,
            images_dict: Dict,
            labels_tensor: Tensor,
            transforms: Optional[albu.Compose] = None
    ):
        self.image_dict = images_dict
        self.transforms = transforms
        self.images_ids = labels_tensor[25,0,:].long().tolist()
        self.labels_tensor = labels_tensor
        self.transform = transforms



    def __len__(self):
        return len(self.images_ids)


    def __getitem__(self, idx):
        transform = albu.Compose([albu.Resize(height=224, width=224), ToTensorV2()])
        img_id = self.images_ids[idx]
        image = self.image_dict[str(img_id)] / 255



        image = image.squeeze().numpy()
        transformed = transform(image=image)
        image = transformed['image'].unsqueeze(0)
        cls_label = self.labels_tensor[:25, 4:-1, idx]
        reg_label = self.labels_tensor[:25, :4, idx]
        reg_label = reg_label / 100
        hole_label = self.labels_tensor[0,-1, idx]



        return image, (reg_label, cls_label, hole_label)






