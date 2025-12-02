from monai.transforms import (
    Compose,
    Spacingd,
    ToTensord,
)
import torch

from .transformation_utils import (
    LoadNrrd,
    NormalizeAxialImages,
    LoadAxialViewLA,
    LoadRandom2DPatches,
    Load2DPatchesPseudoBags,
)


def get_transforms(CONFIG):
    train_transform = Compose(
            [
                LoadNrrd(keys=["image", "la_label"]),  # Load the nii.gz file
                Spacingd(
                    keys=["image", "la_label"],
                    pixdim=(CONFIG['spacing'][0], CONFIG['spacing']
                            [1], CONFIG['spacing'][2]),
                    mode=("trilinear", "nearest"),
                ),
                # Load the axial view of the image and label
                LoadAxialViewLA(keys=["image", "la_label"]),
                # https://docs.monai.io/en/latest/transforms.html#normalizeintensity
                NormalizeAxialImages(keys=["axial_image"], nonzero=True),
                LoadRandom2DPatches(keys=["axial_image", "la_label"], n_patches=CONFIG['n_patches'], patch_size=CONFIG['patch_size'],
                                                no_of_pseudo_bags=CONFIG['no_of_pseudo_bags']),  # Load the patches from the image and label
                ToTensord(keys=["labels"], dtype=torch.float32)
            ]
        )

    val_transform = Compose(
            [
                LoadNrrd(keys=["image", "la_label"]),  # Load the nii.gz file
                Spacingd(
                    keys=["image", "la_label"],
                    pixdim=(CONFIG['spacing'][0], CONFIG['spacing']
                            [1], CONFIG['spacing'][2]),
                    mode=("trilinear", "nearest"),
                ),
                # Load the axial view of the image and label
                LoadAxialViewLA(keys=["image", "la_label"]),
                # https://docs.monai.io/en/latest/transforms.html#normalizeintensity
                NormalizeAxialImages(keys=["axial_image"], nonzero=True),
                # Load the patches from the image and label
                Load2DPatchesPseudoBags(
                    keys=["axial_image", "la_label"], stride=CONFIG['stride'], patch_size=CONFIG['patch_size']),
                ToTensord(keys=["labels"], dtype=torch.float32)
            ]
        )
    return train_transform, val_transform