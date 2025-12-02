from comet_ml import Experiment
import argparse
import glob
import os
import random

import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch

from config import CONFIG, MODEL_NAME, config, disable_comet
from model.qc_model import AttentionMILPseudoBagTier1, AttentionMILPseudoBagTier2
from scripts.qc_dataset import get_patient_records_monai, get_qc_scores
from scripts.qc_trainer import AFibQCAttentionMILPsuedoBagsTrainer
from monai.data import CacheDataset, Dataset

from util.data_transforms_config import get_transforms
from util.early_stopping import EarlyStopping

os.environ['PYTHONHASHSEED'] = str(42)
torch.cuda.manual_seed_all(CONFIG['seed'])
random.seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
torch.cuda.manual_seed(CONFIG['seed'])


def parse_arguments():
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Run model training and evaluation")
    parser.add_argument("--learning_rate", type=float, default=CONFIG['learning_rate'], help="Override learning rate if applicable")
    parser.add_argument("--batch_size", type=int, default=CONFIG['batch_size'], help="Override batch size if applicable")
    parser.add_argument("--epochs", type=int, default=CONFIG['epochs'], help="Override number of epochs if applicable")
    parser.add_argument("--no_of_pseudo_bags", type=int, default=CONFIG['no_of_pseudo_bags'], help="Override no_of_pseudo_bags value if applicable")
    parser.add_argument("--n_patches", type=int, default=CONFIG['n_patches'], 
                        help="No of patches to extract from each image")

    return parser.parse_args()

def update_config(args):
    # List of parameters to update
    params_to_update = [
        "learning_rate",
        "no_of_pseudo_bags",
        "n_patches",
        "batch_size",
        "epochs",
    ]

    for param in params_to_update:
        arg_value = getattr(args, param)
        if arg_value is not None and param in CONFIG:
            CONFIG[param] = arg_value
            print(f"Overriding {param} to {arg_value}")


def main(train_transform, val_transform):
    saved_model_name = CONFIG['saved_model_name']

    data_files = glob.glob(CONFIG['data_path'] + '/*')
    data_files.sort()

    # Get labeled data for k-fold split
    qc_scores, labeled_data_files, qc_dict_scores = get_qc_scores(data_files, CONFIG['qc_label_dict'])
    
    # Initialize StratifiedKFold
    skf = StratifiedKFold(n_splits=CONFIG.get('k_folds', 5), shuffle=True, random_state=CONFIG['seed'])
    
    qc_scores_binary = [0.0 if score <= 2.0 else 1.0 for score in qc_scores]
    
    for fold, (train_val_indices, test_indices) in enumerate(skf.split(labeled_data_files, qc_scores_binary)):
        print(f"\n{'='*50}")
        print(f"Starting Fold {fold + 1}/{CONFIG.get('k_folds', 5)}")
        print(f"{'='*50}")
        
        # Split data based on indices
        train_val_files = [labeled_data_files[i] for i in train_val_indices]
        test_files = [labeled_data_files[i] for i in test_indices]
        
        # Get QC scores for train_val split
        train_val_qc_scores = [qc_scores[i] for i in train_val_indices]
        train_val_qc_scores_binary = [0.0 if score <= 2.0 else 1.0 for score in train_val_qc_scores]
        
        # Further split train_val into train and validation
        train_files, val_files = train_test_split(
            train_val_files, test_size=0.2, random_state=CONFIG['seed'], stratify=train_val_qc_scores_binary
        )
        
        # Calculate class weights for this fold
        train_file_labels = []
        for train_file in train_files:
            train_file_name = train_file.split("/")[-1]
            train_file_labels.append(qc_dict_scores[train_file_name]["label"]["quality_for_fibrosis_assessment"]
            )
            
        class_sample_counts = np.bincount(train_file_labels)
        train_class_weight = 1.0 / class_sample_counts
        samples_train_weights = np.array(
            [train_class_weight[int(label)] for label in train_file_labels]
        )
        
        print(f"Fold {fold + 1} - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
        
        # Update saved model name for this fold
        if isinstance(saved_model_name, list):
            fold_saved_model_name = [name.replace('.pth', f'_fold{fold+1}.pth') for name in saved_model_name]
        else:
            fold_saved_model_name = saved_model_name.replace('.pth', f'_fold{fold+1}.pth')

        train_patient_records = get_patient_records_monai(train_files, data_category='train', qc_dict_json=qc_dict_scores)
        val_patient_records = get_patient_records_monai(val_files, data_category='val', qc_dict_json=qc_dict_scores)
        test_patient_records = get_patient_records_monai(test_files, data_category='test', qc_dict_json=qc_dict_scores)
        
        AFibQCDataset_train = CacheDataset(data=train_patient_records, transform=train_transform, cache_rate=1.0, num_workers=18, copy_cache=False)
        AFibQCDataset_val = CacheDataset(data=val_patient_records, transform=val_transform, cache_rate=1.0, num_workers=8, copy_cache=False)
        AFibQCDataset_test = CacheDataset(data=test_patient_records, transform=val_transform, cache_rate=1.0, num_workers=8, copy_cache=False)
        
        print(f"AFib dataset created")

        print(f"Model saved at: {CONFIG['model_path'] + f'/{fold_saved_model_name}'}")
        early_stopping = EarlyStopping(patience=CONFIG['training_patience'], verbose=False, delta=0.001, path=[CONFIG['model_path'] + f'/{fold_saved_model_name[0]}',
                                                                                                            CONFIG['model_path'] + f'/{fold_saved_model_name[1]}'], score_name='auroc',
                                        start_epoch=0)

        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("PROJECT_NAME"),
            workspace=os.getenv("COMET_WORKSPACE"),
            log_code=True,
            disabled=disable_comet,
        )
        experiment.set_name(f"{MODEL_NAME}_fold_{fold+1}")
        experiment.log_parameter("fold", fold+1)

        experiment.log_code(file_name="model/qc_model.py")
        experiment.log_code(file_name="scripts/qc_trainer.py")
        experiment.log_code(file_name="util/data_transforms_config.py")
        experiment.log_code(file_name="util/transformation_utils.py")
        experiment.log_code(file_name="scripts/qc_dataset.py")

        hyper_params = {"seed": CONFIG['seed'],
                "patch_size": CONFIG['patch_size'],
                "n_patches": CONFIG['n_patches'], # "n_patches": "Number of patches to extract from each image.
                "stride": CONFIG['stride'],
                "enlarge_xy": CONFIG['enlarge_xy'],
                "batch_size": CONFIG['batch_size'], 
                "num_epochs": CONFIG['epochs'],
                "spacing": CONFIG['spacing'],
                "learning_rate": CONFIG['learning_rate'],
                "weight_decay": CONFIG['weight_decay'],
                "no_of_pseudo_bags": CONFIG['no_of_pseudo_bags'],
        }

        experiment.log_parameters(hyper_params)

        tier2_model = AttentionMILPseudoBagTier2(num_classes=1)
        tier1_model = AttentionMILPseudoBagTier1(encoder_name='resnet', n_input_channels=1, num_classes=1, spatial_dims=2)

        afib_qc_model = [tier1_model, tier2_model]

        afib_trainer = AFibQCAttentionMILPsuedoBagsTrainer(tier_1_model=afib_qc_model[0], tier_2_model=afib_qc_model[1], 
                                    train_dataset=AFibQCDataset_train, val_dataset=AFibQCDataset_val, test_dataset=AFibQCDataset_test, 
                                    batch_size=CONFIG['batch_size'], epochs=CONFIG['epochs'], lr=CONFIG['learning_rate'],
                                    train_class_weight=None, experiment=experiment)
        
        print(f"AFib trainer created for {MODEL_NAME}")
        print("Training started")
        
        afib_trainer.train(early_stopping=early_stopping, train_weights=samples_train_weights, class_sample_counts=class_sample_counts)

        print(f"Loading the saved model: {CONFIG['model_path']}/{fold_saved_model_name}")
        afib_trainer.test(model_save_path=fold_saved_model_name, fold=fold)

        print("Testing finished")

        experiment.end()
        print(f"Fold {fold + 1} completed")


if __name__ == '__main__':
    args = parse_arguments()
    update_config(args)

    train_transform, val_transform = get_transforms(CONFIG)

    main(train_transform=train_transform, val_transform=val_transform)