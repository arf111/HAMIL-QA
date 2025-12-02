import torch

gpu_id = 0
machine='lorem_epsum'
seed = 0

n_patches = 60
enlarge_xy = 30

disable_comet = True

no_of_pseudo_bags = 6  # 18 is the lowest number of Axial slices in the dataset
patch_size_2d = (64, 64)

def calculate_stride(patch_size, overlap_percentage):
    return tuple(int(size * (1 - overlap_percentage / 100)) for size in patch_size)

if n_patches % no_of_pseudo_bags != 0:
    raise ValueError("n_patches must be divisible by no_of_pseudo_bags")

overlap_percentage = 50
stride_2d = calculate_stride(patch_size_2d, overlap_percentage)

config = {
    "AFibQCAttentionMILPsuedoBagsNet": {
        'patch_size': patch_size_2d,
        'stride': stride_2d,
        'enlarge_xy': enlarge_xy,
        'n_patches': n_patches,
        'spatial_dims': 2,
        'batch_size': 2,
        'no_of_pseudo_bags': no_of_pseudo_bags,
        'epochs': 50,
        'training_patience': 8,
        'max_clip_grad': 5.0,
        'seed': seed,
        'learning_rate': 1e-4,
        'spacing': [1.5, 1.5, 1.5],
        'weight_decay': 3e-2,
        'data_path': '/home/arefeen_sci/Projects/Left_Atrium_QC/dataset/afib_db',
        'qc_label_dict': '/home/arefeen_sci/Projects/Left_Atrium_QC/dataset/new_surface_area.json',
        'model_path': 'model/saved_models',
        'test_size': 0.2,
        'tier1_saved_model_name': f'afib_qc_attn_pseudo_bags_tier1_baseline_2d_{no_of_pseudo_bags}_{n_patches}_{gpu_id}_{machine}.pth',
        'tier2_saved_model_name': f'afib_qc_attn_pseudo_bags_tier2_baseline_2d_{no_of_pseudo_bags}_{n_patches}_{gpu_id}_{machine}.pth',
        'saved_model_name': [f'afib_qc_attn_pseudo_bags_tier1_baseline_2d_{no_of_pseudo_bags}_{n_patches}_{gpu_id}_{machine}.pth',
                                        f'afib_qc_attn_pseudo_bags_tier2_baseline_2d_{no_of_pseudo_bags}_{n_patches}_{gpu_id}_{machine}.pth']
    },
}

device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

MODEL_NAME = 'AFibQCAttentionMILPsuedoBagsNet'
CONFIG = config[MODEL_NAME]
