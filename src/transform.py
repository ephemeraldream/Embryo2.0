
import albumentations as albu


def get_train_transformations(img_width: int, img_height: int) -> albu.Compose:
    return albu.Compose([
        albu.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, val_shift_limit=10, p=0.1),
        albu.ElasticTransform(
            alpha=1.0,
            sigma=15.0,
            alpha_affine=15,
            interpolation=1,
            border_mode=2,
            p=0.07,
        ),
        albu.CoarseDropout(
            always_apply=False,
            p=0.1,
            max_holes=12,
            max_height=25,
            max_width=25,
            min_holes=7,
            min_height=5,
            min_width=5,
            fill_value=(0, 0, 0),
            mask_fill_value=None,
        ),
    ])

def get_basic_transform(img_width: int, img_height) -> albu.Compose:
    return albu.Compose([albu.Resize(img_width, img_height)])

