import albumentations as albu
from albumentations.pytorch import ToTensorV2

def transform() -> albu.Compose:
    return albu.Compose(
        [
            albu.Resize(width=224, height=224),
            albu.HorizontalFlip(p=0.5),
            albu.VerticalFlip(p=0.2),
            # albu.ElasticTransform(
            #     alpha=1.0,
            #     sigma=15.0,
            #     alpha_affine=20,
            #     interpolation=1,
            #     border_mode=2,
            #     p=0.4,
   # ),
            albu.CoarseDropout(
                always_apply=False,
                p=0.3,
                max_holes=12,
                max_height=40,
                max_width=40,
                min_holes=7,
                min_height=30,
                min_width=30,
                fill_value=2,
                mask_fill_value=None,
            ),
            albu.PixelDropout(
                always_apply=False,
                p=0.3,
                dropout_prob=0.2,
                per_channel=0,
                drop_value=0,
                mask_drop_value=None,
            ),
            albu.Perspective(
                scale=(0.05, 0.1),
                keep_size=True,
                interpolation=1,
                p=0.5,
            ),
            albu.PiecewiseAffine(
                always_apply=False,
                p=0.1,
                scale=(0, 0.03),
                nb_rows=(4, 4),
                nb_cols=(4, 4),
                interpolation=0,
                mask_interpolation=0,
                cval=0,
                cval_mask=0,
                mode='constant',
                absolute_scale=0,
                keypoints_threshold=0.01,
            ),

            albu.Rotate(limit=25),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            albu.GaussianBlur(p=0.3),
            albu.GaussNoise(
                always_apply=False,
                p=0.5,
                var_limit=(10.0, 50.0),
                per_channel=True,
                mean=0,
            ),
            albu.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
            albu.RandomResizedCrop(
                scale=(0.8, 1.0),
                height=224,
                width=224,
                always_apply=True,
            ),
            ToTensorV2()
        ],
    )


def augment_image(image):
    augmented = transform(image=image)
    return augmented['image']
