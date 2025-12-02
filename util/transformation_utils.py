import numpy as np
import nrrd

from collections.abc import Hashable, Mapping

from monai.config import KeysCollection
import torch
from config import CONFIG
from monai.transforms import NormalizeIntensity
from monai.transforms.transform import MapTransform

from PIL import Image

def load_nrrd_data(file_path):
    """Load the NRRD file and return the data array."""
    data, _ = nrrd.read(file_path)
    data = np.expand_dims(data, axis=0)  # Add channel dimension

    return data

def crop_random_2d_patches(volume, patch_size, n_patches):
    """Crop random patches from the volume."""
    patches = []
    
    assert volume.shape[0] >= patch_size[0], f"Patch size cannot be larger than the volume size, volume shape: {volume.shape}, patch size: {patch_size}"
    assert volume.shape[1] >= patch_size[1], f"Patch size cannot be larger than the volume size, volume shape: {volume.shape}, patch size: {patch_size}"

    for _ in range(n_patches):
        # volume.shape[0] = 256, patch_size[0] = 64 => 0, 192
        x = np.random.randint(0, volume.shape[0] - patch_size[0] + 1)
        # volume.shape[1] = 256, patch_size[1] = 64 => 0, 192
        y = np.random.randint(0, volume.shape[1] - patch_size[1] + 1)

        patch = volume[x:x + patch_size[0], y:y + patch_size[1]]

        assert patch.shape == patch_size, f"Patch shape is {patch.shape} instead of {patch_size}"
        patch = np.expand_dims(patch, axis=0)  # Add channel dimension

        patches.append(patch)

    if len(patches) == 0:
        raise ValueError("No patches were generated")

    patches = np.stack(patches, axis=0)  # (n_patches, 64, 64, 7)

    return patches

def crop_patches_2d(volume, patch_size, stride):
    """Crop overlapping patches from the volume."""
    patches = []

    patches_position = []

    for x in range(0, volume.shape[0] - patch_size[0] + 1, stride[0]):  # 0, 64, 32
        for y in range(0, volume.shape[1] - patch_size[1] + 1, stride[1]):
            patch = volume[x:x + patch_size[0], y:y + patch_size[1]]
            patch = np.expand_dims(patch, axis=0) # Add channel dimension
            patches_position.append((x, y)) # (n_patches, 2)
            patches.append(patch)

    if len(patches) == 0:
        raise ValueError("No patches were generated")

    patches_position = np.stack(patches_position, axis=0) # (n_patches, 2)

    patches = np.stack(patches, axis=0)  # (n_patches, 64, 64, 7)

    return patches, patches_position

def find_bounding_box_2d(label_data):
    """Find the bounding box for the blood pool regions."""
    xs, ys = np.where(label_data == 1)
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    return x_min, x_max, y_min, y_max


def extract_and_crop_random_2d_patches(mri_data, label_data, patch_size=(64, 64), n_patches=60):
    # shape of label_data:
    # Find the bounding box for the blood pool
    x_min, x_max, y_min, y_max = find_bounding_box_2d(label_data)

    squeezed_mri_data = mri_data
    # Expand x, y dimensions to make the bounding box larger
    x_min = max(0, x_min - CONFIG['enlarge_xy'])
    x_max = min(squeezed_mri_data.shape[0] - 1, x_max + CONFIG['enlarge_xy'])

    y_min = max(0, y_min - CONFIG['enlarge_xy'] - 20)
    y_max = min(squeezed_mri_data.shape[1] - 1, y_max + CONFIG['enlarge_xy'] + 20)

    # Extract the sub-volume containing the blood pool
    sub_mri_data = squeezed_mri_data[x_min:x_max +1, y_min:y_max+1] # (192, 192)

    # Crop random patches from the sub-volume
    patches = crop_random_2d_patches(sub_mri_data, patch_size, n_patches)

    return patches, sub_mri_data

def extract_and_crop_patches_2d(mri_data, label_data, patch_size=(64, 64), stride=(32, 32), enlarge_xy=20):
    # shape of label_data:
    # Find the bounding box for the blood pool
    x_min, x_max, y_min, y_max = find_bounding_box_2d(label_data)

    squeezed_mri_data = mri_data
    # Expand x, y dimensions to make the bounding box larger
    x_min = max(0, x_min - enlarge_xy)
    x_max = min(squeezed_mri_data.shape[0] - 1, x_max + enlarge_xy)

    y_min = max(0, y_min - enlarge_xy - 20)
    y_max = min(squeezed_mri_data.shape[1] - 1, y_max + enlarge_xy + 20)

    # Extract the sub-volume containing the blood pool
    sub_mri_data = squeezed_mri_data[x_min:x_max +1, y_min:y_max+1] # (192, 192)

    # Crop patches from the sub-volume
    patches, patches_position = crop_patches_2d(sub_mri_data, patch_size, stride=stride)

    return patches, sub_mri_data, patches_position


class LoadNrrd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(
        self, data: Mapping[Hashable, torch.Tensor]
    ) -> dict[Hashable, torch.Tensor]:
        d = dict(data)

        for keys in self.keys:
            d[keys] = load_nrrd_data(d[keys])

        return d
    
class LoadAxialViewLA(MapTransform):
    def __init__(
        self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None
    ) -> None:
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        
        d = dict(data)
        
        mri_data = d['image'] # (1, 254, 190, 36)
        label_data = d['la_label']
        
        axial_images, axial_labels = [], []

        for i in range(mri_data.shape[3]):
            
            if label_data[0, :, :, i].sum() == 0:
                continue

            axial_images.append(mri_data[0, :, :, i].transpose(1, 0))
            axial_labels.append(label_data[0, :, :, i].transpose(1, 0))

        axial_images = np.stack(axial_images, axis=0) # (30, 256, 190)
        axial_labels = np.stack(axial_labels, axis=0) # (30, 256, 190)

        d['axial_image'] = axial_images # (30, 256, 190)
        d['la_label'] = axial_labels # (30, 256, 190)
        
        return d
    
class NormalizeAxialImages(MapTransform):
        def __init__(
            self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, nonzero=False, channel_wise=False, rgb_image=False,
        ) -> None:
            super().__init__(keys, allow_missing_keys)
            self.rgb_image = rgb_image
            if rgb_image:
                self.normalize_intensity = NormalizeIntensity(nonzero=nonzero, channel_wise=channel_wise, subtrahend=[0.485, 0.456, 0.406], divisor=[0.2290, 0.2240, 0.2250])
            else:
                self.normalize_intensity = NormalizeIntensity(nonzero=nonzero, channel_wise=channel_wise)

        def to_rgb_pil(self, image_tensor):
            # image_tensor: torch.Tensor, shape (H, W)
            # Convert to numpy, scale to 0-255, uint8, then to PIL and RGB
            arr = image_tensor.cpu().numpy()
            arr = arr.astype('uint8')
            pil_img = Image.fromarray(arr, mode='L').convert('RGB')
            return pil_img

        def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
    
            d = dict(data)

            axial_images = d['axial_image']

            if self.rgb_image:
                d['axial_image'] = torch.stack([
                    self.normalize_intensity(
                        torch.from_numpy(np.array(self.to_rgb_pil(image.squeeze(0)))).permute(2, 0, 1).float()
                    ) for image in axial_images
                ])
            else:  # Shape is (N, C, H, W)
                normalized_images = []
                for image in axial_images:  # image shape: (C, H, W)
                    normalized_image = self.normalize_intensity(image)
                    normalized_images.append(normalized_image)

                d['axial_image'] = torch.stack(normalized_images)

            return d

class Load2DPatchesPseudoBags(MapTransform):
    def __init__(
        self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None, stride=None, patch_size=(64, 64)
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.stride = stride
        self.patch_size = patch_size

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        
        d = dict(data)
        
        axial_images = d['axial_image'] # shape: (30, 256, 190), (slices, height, width)
        axial_labels = d['la_label'] # shape: (30, 256, 190), (slices, height, width)

        pseudo_bags = []

        for i in range(axial_images.shape[0]):
            patch, _, _ = extract_and_crop_patches_2d(mri_data=axial_images[i], label_data=axial_labels[i],
                                                patch_size=self.patch_size, stride=self.stride, enlarge_xy=CONFIG['enlarge_xy'])
            
            pseudo_bags.append(torch.tensor(patch, dtype=torch.float32))

        # Combine all patches into a single list
        all_patches = [patch for bag in pseudo_bags for patch in bag]

        # Calculate the number of patches per pseudo-bag
        n_patches = len(all_patches)
        patches_per_bag = n_patches // CONFIG['no_of_pseudo_bags']
        remainder = n_patches % CONFIG['no_of_pseudo_bags']

        # Distribute patches into pseudo-bags
        pseudo_bags = []

        start = 0
        for i in range(CONFIG['no_of_pseudo_bags']):
            end = start + patches_per_bag
            if i < remainder:
                end += 1

            pseudo_bags.append(torch.stack(all_patches[start:end]))

            start = end

        d['pseudo_bags'] = pseudo_bags

        del d['la_label']
        del d['axial_image']

        return d

class LoadRandom2DPatches(MapTransform):
    def __init__(
        self, keys: KeysCollection, strict_check: bool = True, allow_missing_keys: bool = False, channel_dim=None, n_patches=60, patch_size=(64, 64), no_of_pseudo_bags=10,
    ) -> None:
        super().__init__(keys, allow_missing_keys)

        self.n_patches = n_patches
        self.patch_size = patch_size
        self.no_of_pseudo_bags = no_of_pseudo_bags

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> dict[Hashable, torch.Tensor]:
        
        d = dict(data)
        
        axial_images = d['axial_image'] # shape: (30, 256, 190), (slices, height, width)
        axial_labels = d['la_label'] # shape: (30, 256, 190), (slices, height, width)

        pseudo_bags = []

        no_of_patches = 0

        assert axial_images.shape[0] >= self.no_of_pseudo_bags, f"Number of slices is {axial_images.shape[0]} which is less than the number of pseudo-bags {self.no_of_pseudo_bags}"

        # Randomly select no of pseudo-bags from the axial images
        random_indices = np.random.choice(a=axial_images.shape[0], size=self.no_of_pseudo_bags, replace=False)

        pseudo_bags_slices = axial_images[random_indices] # shape: (no_of_pseudo_bags, 256, 190)
        pseudo_bags_slices_labels = axial_labels[random_indices] # shape: (no_of_pseudo_bags, 256, 190)

        # divide n_patches into no_of_pseudo_bags, remainder will be randomly picked from the axial images
        n_instances_per_bag = self.n_patches // self.no_of_pseudo_bags # 100 // 15 = 6
        remainder = self.n_patches % self.no_of_pseudo_bags # 100 % 15 = 10

        for i in range(self.no_of_pseudo_bags): # Loop over the number of pseudo-bags
            patches_per_bag, _ = extract_and_crop_random_2d_patches(mri_data=pseudo_bags_slices[i], label_data=pseudo_bags_slices_labels[i],
                                                patch_size=self.patch_size, n_patches=n_instances_per_bag)
            
            no_of_patches += patches_per_bag.shape[0]
            pseudo_bags.append(patches_per_bag) 
        
        # select remainder number of patches from the pseudo-bags randomly.
        if remainder > 0:
            random_indices_remainder = np.random.choice(a=self.no_of_pseudo_bags, size=remainder, replace=False) # Select remainder number of pseudo-bags, since no_of_pseudo_bags > remainder

            pseudo_bags_slices_remainder = pseudo_bags_slices[random_indices_remainder] # shape: (remainder, 256, 190)
            pseudo_bags_slices_labels_remainder = pseudo_bags_slices_labels[random_indices_remainder] # shape: (remainder, 256, 190)

            for i in range(remainder):
                patch_per_bag, _ = extract_and_crop_random_2d_patches(mri_data=pseudo_bags_slices_remainder[i], label_data=pseudo_bags_slices_labels_remainder[i],
                                                    patch_size=self.patch_size, n_patches=1) # n_patches=1

                no_of_patches += patch_per_bag.shape[0]
                pseudo_bags[random_indices_remainder[i]] = np.concatenate((pseudo_bags[random_indices_remainder[i]], patch_per_bag), axis=0)

        assert no_of_patches == self.n_patches, f"Number of patches is {no_of_patches} instead of {self.n_patches}"
        assert len(pseudo_bags) == self.no_of_pseudo_bags, f"Length of pseudo-bags is {len(pseudo_bags)} instead of {self.no_of_pseudo_bags}"

        d['pseudo_bags'] = pseudo_bags #  len(pseudo_bags) = no_of_pseudo_bags, each element shape = (n_patches_per_bag, 1, 60, 60)

        del d['axial_image']
        del d['la_label']

        return d