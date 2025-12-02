from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
import os
import json

from scripts import base_dir

from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.data import ThreadDataLoader, decollate_batch
from monai.transforms import Compose, Activations
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score, BinaryConfusionMatrix, BinarySpecificity, BinaryAUROC, BinaryROC, BinaryPrecisionRecallCurve
from tqdm import tqdm

from config import CONFIG, device, gpu_id

from PIL import Image

class AFibQCAttentionMILPsuedoBagsTrainer:
    def __init__(self, tier_1_model, tier_2_model, train_dataset, val_dataset, test_dataset, batch_size, epochs, lr, 
                 train_class_weight=None, experiment=None):
        self.tier_1_model = tier_1_model
        self.tier_2_model = tier_2_model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr

        self.experiment = experiment

        self.device = device
        self.tier_1_model.to(self.device)
        self.tier_2_model.to(self.device)

        self.models = [self.tier_1_model, self.tier_2_model]
        
        self.optimizer2 = optim.Adam([
                {'params': self.tier_1_model.parameters(), 'lr': self.lr * 0.1},
                {'params': self.tier_2_model.parameters(), 'lr': self.lr}
        ], weight_decay=CONFIG['weight_decay'])
        
        self.bce_logits_loss = nn.BCEWithLogitsLoss()
        train_class_weight = torch.tensor(train_class_weight, dtype=torch.float32).to(self.device) if train_class_weight is not None else None
        self.ce_loss = nn.CrossEntropyLoss(weight=train_class_weight)
        
        self.post_process_binary = Compose([Activations(sigmoid=True)])
        self.post_process_category = Compose([Activations(softmax=True)])

        self.ac, self.f1, self.spec, self.cm, self.auroc, self.roc, self.prcurve = self.get_classification_metrics()
    
        self.scheduler2 = optim.lr_scheduler.CosineAnnealingLR(self.optimizer2, T_max=epochs, eta_min=1e-7)

    def get_classification_metrics(self):
        return BinaryAccuracy(threshold=0.5).to(self.device), BinaryF1Score(threshold=0.5).to(self.device), BinarySpecificity(threshold=0.5).to(self.device), \
                BinaryConfusionMatrix(threshold=0.5).to(self.device), BinaryAUROC(thresholds=5).to(self.device), BinaryROC(thresholds=5).to(self.device), BinaryPrecisionRecallCurve(thresholds=5).to(self.device)
        
        
    def train_one_supervised_epoch(self, train_loader, epoch_number):
        self.tier_1_model.train()
        self.tier_2_model.train()

        running_loss = 0.0
        predicted_logits_of_overall_train, true_labels_of_overall_train = [], []
        tier1_losses, tier2_losses = 0, 0

        epoch_iterator = tqdm(train_loader, desc=f'Phase: train', total=len(train_loader), unit='batch', dynamic_ncols=True)

        iterat = 0
        for batch in epoch_iterator:
            inputs, labels = batch['pseudo_bags'], batch['labels'] # inputs (list) length: no_of_pseudo_bags, inputs[0] shape: (batch_size=1, patches_per_bag, 1, 64, 64, 7)
            inputs = inputs.to(self.device)

            labels = {key: value.to(self.device).unsqueeze(1) for key, value in labels.items()}
            
            pseudo_bag_preds, pseudo_bag_labels, pseudo_bags_instance_distilled_features = [], [], []
            
            with torch.set_grad_enabled(True):
                pseudo_bag_preds, embedding_attention_weights, instance_logits, weighted_feature_space, feature_space = self.tier_1_model(inputs)
                pseudo_bag_labels = labels['quality_for_fibrosis_assessment'].unsqueeze(1).expand_as(pseudo_bag_preds) # pseudo_bag_labels shape: (no_of_pseudo_bags, 1)
   
                feature_space = torch.mean(feature_space, dim=2) # feature_space shape: (no_of_pseudo_bags, L)

                overall_embedding_logits, _ = self.tier_2_model(feature_space) # overall_embedding_logits shape: (1, 1)

                tier1_loss = self.bce_logits_loss(input=pseudo_bag_preds, target=pseudo_bag_labels) # pseudo_bag_preds shape: (no_of_pseudo_bags, 1), pseudo_bag_labels shape: (no_of_pseudo_bags, 1)
                tier2_loss = self.bce_logits_loss(input=overall_embedding_logits, target=labels['quality_for_fibrosis_assessment']) # overall_embedding_logits shape: (1, 1)
            
                self.optimizer2.zero_grad()
            
                loss = tier1_loss + tier2_loss

                loss.backward()

                self.optimizer2.step()

            predicted_overall = [self.post_process_binary(i) for i in decollate_batch(overall_embedding_logits)]

            predicted_logits_of_overall_train.extend(predicted_overall)
            true_labels_of_overall_train.extend(labels['quality_for_fibrosis_assessment'].int())

            tier1_losses += tier1_loss.item() * pseudo_bag_labels.size(1)
            tier2_losses += tier2_loss.item()
            running_loss += tier1_loss.item() * pseudo_bag_labels.size(1) + tier2_loss.item()

        self.scheduler2.step()

        predicted_logits_of_overall_train = torch.stack(predicted_logits_of_overall_train)
        true_labels_of_overall_train = torch.stack(true_labels_of_overall_train)
        accuracy = self.ac(predicted_logits_of_overall_train, true_labels_of_overall_train)
        f1_score = self.f1(predicted_logits_of_overall_train, true_labels_of_overall_train)
        auroc = self.auroc(predicted_logits_of_overall_train, true_labels_of_overall_train)

        epoch_loss = running_loss / len(train_loader)
        tier1_losses = tier1_losses / (len(train_loader) * CONFIG['no_of_pseudo_bags'])
        tier2_losses = tier2_losses / len(train_loader)

        self.experiment.log_metrics({"Train Tier1 Loss": tier1_losses,
                                "Train Tier2 Loss": tier2_losses,
                                "Train Accuracy": accuracy,
                                "Train F1 Score": f1_score,
                                "Train AUROC": auroc,}, step=epoch_number)

        print(f'Train Tier1 Loss: {tier1_losses:.4f}, Tier2 Loss: {tier2_losses:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1_score:.4f}, AUROC: {auroc:.4f}')

        return epoch_loss

    def test_one_supervised_epoch(self, test_loader, epoch_number, category='val', fold=None):
        self.tier_1_model.eval()
        self.tier_2_model.eval()

        tier1_losses, tier2_losses = 0, 0
        running_loss = 0.0

        predicted_logits_of_overall_test, true_labels_of_overall_test = [], []

        iterat = 0
        for batch in tqdm(test_loader, desc=f'Phase: {category}', total=len(test_loader), unit='batch', dynamic_ncols=True):
            inputs, labels = batch['pseudo_bags'], batch['labels']
            
            inputs = [i.squeeze(0) for i in inputs]
            inputs = [i.to(self.device) for i in inputs]

            labels = {key: value.to(self.device).unsqueeze(1) for key, value in labels.items()}
            
            pseudo_bag_preds, pseudo_bag_labels, pseudo_bags_instance_distilled_features = [], [], []

            with torch.no_grad():
                
                for i, bag in enumerate(inputs):
                    bag_input = bag.to(self.device)

                    embedding_logits, embedding_attention_weights, instance_logits, weighted_feature_space, feature_space = self.tier_1_model(bag_input)

                    pseudo_bag_preds.append(embedding_logits.squeeze(1)), pseudo_bag_labels.append(labels['quality_for_fibrosis_assessment'].squeeze(1))
                    
                    pseudo_bags_instance_distilled_features.append(torch.mean(feature_space, dim=1))
                   
                pseudo_bag_preds = torch.stack(pseudo_bag_preds) # pseudo_bag_preds shape for binary: (no_of_pseudo_bags, 1), for multiclass: (no_of_pseudo_bags, 5)
                pseudo_bag_labels = torch.stack(pseudo_bag_labels) # pseudo_bag_labels shape for binary: (no_of_pseudo_bags, 1), for multiclass: (no_of_pseudo_bags, 5)
                
                pseudo_bags_instance_distilled_features = torch.stack(pseudo_bags_instance_distilled_features).squeeze() # shape: (no_of_pseudo_bags, L)

                overall_embedding_logits, overall_embedding_attention_weights = self.tier_2_model(pseudo_bags_instance_distilled_features)

                tier1_loss = self.bce_logits_loss(input=pseudo_bag_preds, target=pseudo_bag_labels)
                tier2_loss = self.bce_logits_loss(input=overall_embedding_logits, target=labels['quality_for_fibrosis_assessment'].squeeze(1)) # overall_embedding_logits shape: (1, 1)
            
                tier1_losses += tier1_loss.item() * pseudo_bag_labels.size(0)
                tier2_losses += tier2_loss.item()
            
                running_loss += tier1_loss.item() * pseudo_bag_labels.size(0) + tier2_loss.item()

            predicted_overall = [self.post_process_binary(i).to(self.device) for i in decollate_batch(overall_embedding_logits)]
            
            predicted_logits_of_overall_test.extend(predicted_overall)
            true_labels_of_overall_test.extend(labels['quality_for_fibrosis_assessment'].int())

        epoch_loss = running_loss / len(test_loader)
        tier1_losses = tier1_losses / (len(test_loader) * CONFIG['no_of_pseudo_bags'])
        tier2_losses = tier2_losses / len(test_loader)

        predicted_logits_of_overall_test = torch.stack(predicted_logits_of_overall_test).squeeze()
        true_labels_of_overall_test = torch.stack(true_labels_of_overall_test).squeeze()

        accuracy = self.ac(predicted_logits_of_overall_test, true_labels_of_overall_test)
        auroc = self.auroc(predicted_logits_of_overall_test, true_labels_of_overall_test)
        f1_score = self.f1(predicted_logits_of_overall_test, true_labels_of_overall_test)

        self.experiment.log_metrics({f"{category} Accuracy": accuracy,
                                f"{category} AUROC": auroc,
                                f"{category} F1 Score": f1_score,
                                f"{category} Tier1 Loss": tier1_losses,
                                f"{category} Tier2 Loss": tier2_losses}, step=epoch_number)
        
        if category == 'test':
            confusion_matrix = self.cm(predicted_logits_of_overall_test, true_labels_of_overall_test)
            confusion_matrix = confusion_matrix.cpu().numpy()

            self.experiment.log_confusion_matrix(matrix=confusion_matrix, 
                                            title=f"{category} Confusion Matrix", 
                                            file_name=f"test_supervised_confusion_matrix_fold_{fold}.png")

        print(f'{category} Tier1 Loss: {tier1_losses:.4f}, Tier2 Loss: {tier2_losses:.4f}, Accuracy: {accuracy:.4f}, AUROC: {auroc:.4f}, F1 Score: {f1_score:.4f}')

        return auroc

    def custom_collate_fn(self, batch):
        """
        Custom collate function to format pseudo bags into a batched tensor.

        batch: List of batch_size elements, where each element is a dictionary:
            {
                'pseudo_bags': list of n_pseudo_bags tensors (each tensor: (patches_per_bag, 1, h, w, d)),
                'labels': dict with keys mapping to label tensors
            }

        Returns:
            - stacked_inputs: Tensor of shape (batch_size, n_pseudo_bags, patches_per_bag, 1, h, w, d)
            - stacked_labels: Dict of stacked label tensors
        """
        # Unzip batch into pseudo_bags and labels
        pseudo_bags, labels = zip(*[(sample['pseudo_bags'], sample['labels']) for sample in batch])

        # Convert pseudo_bags (list of list of tensors) into a single batched tensor
        stacked_inputs = torch.stack([torch.stack([torch.tensor(bag) for bag in bags]) for bags in pseudo_bags])  
        # Final shape: (batch_size, n_pseudo_bags, patches_per_bag, 1, 64, 64, 7)

        # Convert labels dict into batched tensors
        stacked_labels = {key: torch.stack([lbl[key].clone().detach() for lbl in labels]) for key in labels[0]}    

        return {'pseudo_bags': stacked_inputs, 'labels': stacked_labels}


    def train(self, early_stopping, train_weights=None, class_sample_counts=None):
        samples_train_weights = torch.from_numpy(train_weights).float()
        self.class_sample_counts = torch.tensor(class_sample_counts).float().to(self.device) if class_sample_counts is not None else None

        sampler = WeightedRandomSampler(weights=samples_train_weights, num_samples=len(samples_train_weights), replacement=True)
        
        self.train_loader = ThreadDataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=0, collate_fn=self.custom_collate_fn, sampler=sampler)
        self.val_loader = ThreadDataLoader(dataset=self.val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

        for epoch in range(self.epochs):
            print(f'Epoch {epoch+1}/{self.epochs}')

            train_loss = self.train_one_supervised_epoch(train_loader=self.train_loader, epoch_number=epoch)
            test_auroc = self.test_one_supervised_epoch(test_loader=self.val_loader, category='val', epoch_number=epoch)

            if epoch >= early_stopping.start_epoch:
                early_stopping(test_auroc, [self.tier_1_model, self.tier_2_model])

                if early_stopping.early_stop:
                    print("Early stopping")
                    break

            print("=====================================")

    def test(self,model_save_path=None, fold=None):
        # test_loader = ThreadDataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        self.train_loader = ThreadDataLoader(dataset=self.train_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        self.val_loader = ThreadDataLoader(dataset=self.val_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
        self.test_loader = ThreadDataLoader(dataset=self.test_dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=0)

        print(f"Loading the tier1 model from {base_dir + '/' + CONFIG['model_path'] + '/' + model_save_path[0]}")
        print(f"Loading the tier2 model from {base_dir + '/' + CONFIG['model_path'] + '/' + model_save_path[1]}")

        self.tier_1_model.load_state_dict(torch.load(base_dir + "/" + CONFIG['model_path'] + "/" + model_save_path[0], weights_only=True))
        self.tier_2_model.load_state_dict(torch.load(base_dir + "/" + CONFIG['model_path'] + "/" + model_save_path[1], weights_only=True))
        self.test_one_supervised_epoch(test_loader=self.test_loader, epoch_number=0, category='test', fold=fold)

    def compute_attention_save(self, model_save_path=None):
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=torch.cuda.is_available(), 
                                 pin_memory_device=f"cuda:{gpu_id}")

        self.tier_1_model.load_state_dict(torch.load(base_dir + "/" + CONFIG['model_path'] + "/" + CONFIG['tier1_saved_model_name']))
        self.tier_2_model.load_state_dict(torch.load(base_dir + "/" + CONFIG['model_path'] + "/" + CONFIG['tier2_saved_model_name']))

        self.tier_1_model.eval()
        self.tier_2_model.eval()

        for batch in test_loader:
            inputs, labels, p_id = batch['pseudo_bags'], batch['labels'], batch['p_id'] # pseudo_bags shape: (no_of_pseudo_bags, patches_per_bag, 1, 64, 64, 7)
            
            inputs = [i.squeeze(0) for i in inputs]
            inputs = [i.to(self.device) for i in inputs]

            labels = {key: value.to(self.device).unsqueeze(1) for key, value in labels.items()}
            pseudo_bag_preds, pseudo_bag_labels, pseudo_bags_instance_distilled_features = [], [], []
            tier1_attention_weights_max_index = []

            with torch.no_grad():
                for i, bag in enumerate(inputs):
                    bag_input = bag.to(self.device)

                    embedding_logits, embedding_attention_weights, instance_logits, weighted_feature_space, feature_space = self.tier_1_model(bag_input)

                    # normalize embedding_attention_weights
                    embedding_attention_weights = torch.softmax(embedding_attention_weights, dim=1) # embedding_attention_weights shape: (1, patches_per_bag)
                    
                    max_attention_index = torch.argmax(embedding_attention_weights, dim=1) # max_attention_index shape: (1)

                    # get embedding_attention_weights for the max_attention_index
                    tier1_attention_weights_max_index.append(max_attention_index) # tier1_attention_weights_max_index shape: (no_of_pseudo_bags, 1)

                    pseudo_bag_preds.append(embedding_logits.squeeze(1)), pseudo_bag_labels.append(labels['quality_for_fibrosis_assessment'].squeeze(1))

                    pseudo_bags_instance_distilled_features.append(torch.mean(feature_space, dim=0))
                    
                pseudo_bag_preds = torch.stack(pseudo_bag_preds)
                pseudo_bag_labels = torch.stack(pseudo_bag_labels)
                pseudo_bags_instance_distilled_features = torch.stack(pseudo_bags_instance_distilled_features) # shape: (no_of_pseudo_bags, L)

                tier1_attention_weights_max_index = torch.stack(tier1_attention_weights_max_index) # tier1_attention_weights_max_index shape: (no_of_pseudo_bags, 1)

                overall_embedding_logits, overall_embedding_attention_weights = self.tier_2_model(pseudo_bags_instance_distilled_features)

                # normalize overall_embedding_attention_weights
                overall_embedding_attention_weights = torch.softmax(overall_embedding_attention_weights, dim=1) # overall_embedding_attention_weights shape: (1, no_of_pseudo_bags)

                # get the max attention weight index
                max_attention_index = torch.argmax(overall_embedding_attention_weights, dim=1) # max_attention_index shape: (1)

                image_3d = batch['original_axial_images'].squeeze(0).to(self.device) # image_3d shape: (slices, height, width)

                axial_view_image = image_3d[max_attention_index] # batch['image'] shape: (slices, height, width), axial_view_image shape: (height, width)

                sub_image_3d = batch['pseudo_bags_original_sub_mri_data']
                sub_image_3d = [i.squeeze(0) for i in sub_image_3d]
                sub_image_3d = [i.to(self.device) for i in sub_image_3d]

                max_attn_sub_image_3d = sub_image_3d[max_attention_index] # max_attn_sub_image_3d shape: (height, width)

                # get the max_attention_patch from original pseudo bags
                original_pseudo_bags = batch['pseudo_bags_with_original'] # original_pseudo_bags shape: (no_of_pseudo_bags, patches_per_bag, 1, 64, 64, 7)
                original_pseudo_bags = [i.squeeze(0) for i in original_pseudo_bags]
                original_pseudo_bags = [i.to(self.device) for i in original_pseudo_bags]

                original_patches_position = batch['original_patches_position'] # len(original_patches_position) = no_of_pseudo_bags, each element shape = (n_patches_per_bag, 2)
                original_patches_position = [i.squeeze(0) for i in original_patches_position]
                original_patches_position = [i.to(self.device) for i in original_patches_position]

                max_attention_patch = original_pseudo_bags[max_attention_index.item()]
                max_attention_patch_position = original_patches_position[max_attention_index.item()] # max_attention_patch_position: (n_patches_per_bag, 2)

                assert original_pseudo_bags[max_attention_index.item()].shape[0] >= tier1_attention_weights_max_index[max_attention_index.item()].item(), f"inputs[max_attention_index.item()].shape[0]: {inputs[max_attention_index.item()].shape[0]}, tier1_attention_weights_max_index[max_attention_index.item()].item(): {tier1_attention_weights_max_index[max_attention_index.item()].item()}"

                max_attention_patch = max_attention_patch[tier1_attention_weights_max_index[max_attention_index.item()]].squeeze(0) # max_attention_patch shape: (1, 60, 60)
                max_attention_patch_position = max_attention_patch_position[tier1_attention_weights_max_index[max_attention_index.item()]] # max_attention_patch_position: (1, 2)

                max_attention_patch_position_x, max_attention_patch_position_y = int(max_attention_patch_position[0][0].item()), \
                                    int(max_attention_patch_position[0][1].item())

                # generate a heatmap on max_attn_sub_image_3d from using max_attention_patch_position. 
                # The heatmap will be in max_attn_sub_image_3d[max_attention_patch_position_x: max_attention_patch_position_x+patch_size, max_attention_patch_position_y: max_attention_patch_position_y+patch_size]
                
                max_attn_sub_image_3d = max_attn_sub_image_3d.squeeze(0).cpu().numpy()

                heatmap = np.zeros_like(max_attn_sub_image_3d)
                heatmap[max_attention_patch_position_x: max_attention_patch_position_x+CONFIG['patch_size'][0],
                        max_attention_patch_position_y: max_attention_patch_position_y+CONFIG['patch_size'][1]] = 1
                
                # save the axial_view_image and max_attention_patch
                axial_view_image = axial_view_image.squeeze(0).cpu().numpy()

                axial_view_image = Image.fromarray(axial_view_image.astype(np.uint8))
                max_attn_sub_image_3d = Image.fromarray(max_attn_sub_image_3d.astype(np.uint8))

                if not os.path.exists(base_dir + '/attention_maps'):
                    os.makedirs(base_dir + '/attention_maps')

                axial_view_image.save(base_dir + f'/attention_maps/axial_view_image_{p_id[0]}.png')
                max_attn_sub_image_3d.save(base_dir + f'/attention_maps/max_attn_sub_image_3d_{p_id[0]}.png')
                                # Apply color to the heatmap (use 'color' for a single color, instead of 'colors')
                plt.imshow(max_attn_sub_image_3d, cmap='gray')
                plt.imshow(heatmap, cmap='jet', alpha=0.5)
                plt.axis('off')
                plt.savefig(base_dir + f'/attention_maps/heatmap_{p_id[0]}.png')
                plt.close()
                # Save the overlaid image
                # overlaid_image.save(base_dir + f'/attention_maps/overlaid_image_{p_id[0]}.png')
                # max_attention_patch.save(base_dir + f'/attention_maps/max_attention_patch_{p_id[0]}.png')

    def get_test_results(self, model_save_path=None):
        # test_loader = ThreadDataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(dataset=self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=8, pin_memory=torch.cuda.is_available(), 
                                 pin_memory_device=f"cuda:{gpu_id}")

        print(f"Loading the tier1 model from {base_dir + '/' + CONFIG['model_path'] + '/' + CONFIG['tier1_saved_model_name']}")
        print(f"Loading the tier2 model from {base_dir + '/' + CONFIG['model_path'] + '/' + CONFIG['tier2_saved_model_name']}")

        self.tier_1_model.load_state_dict(torch.load(base_dir + "/" + CONFIG['model_path'] + "/" + CONFIG['tier1_saved_model_name'], weights_only=True))
        self.tier_2_model.load_state_dict(torch.load(base_dir + "/" + CONFIG['model_path'] + "/" + CONFIG['tier2_saved_model_name'], weights_only=True))
        
        self.tier_1_model.eval()
        self.tier_2_model.eval()
        
        predicted_logits_of_overall_test, p_ids_list = [], []

        for batch in tqdm(test_loader, desc=f'Phase: test', total=len(test_loader), unit='batch', dynamic_ncols=True):
        # for batch in test_loader:
            inputs, p_id = batch['pseudo_bags'], batch['p_id']
            
            inputs = [i.squeeze(0) for i in inputs]
            inputs = [i.to(self.device) for i in inputs]
            
            pseudo_bag_preds, pseudo_bags_instance_distilled_features = [], []

            with torch.no_grad():            
                for i, bag in enumerate(inputs):
                    bag_input = bag.to(self.device)

                    embedding_logits, embedding_attention_weights, instance_logits, weighted_feature_space, feature_space = self.tier_1_model(bag_input)

                    # embedding_attention_weights shape: (1, patches_per_bag)
                    
                    pseudo_bag_preds.append(embedding_logits.squeeze(1))
                        
                    pseudo_bags_instance_distilled_features.append(torch.mean(feature_space, dim=0))
                    
                pseudo_bag_preds = torch.stack(pseudo_bag_preds) # pseudo_bag_preds shape for binary: (no_of_pseudo_bags, 1), for multiclass: (no_of_pseudo_bags, 5)
                
                pseudo_bags_instance_distilled_features = torch.stack(pseudo_bags_instance_distilled_features) # shape: (no_of_pseudo_bags, L)

                overall_embedding_logits, overall_embedding_attention_weights = self.tier_2_model(pseudo_bags_instance_distilled_features)

                # overall_embedding_logits shape: (1, 1), overall_embedding_attention_weights shape: (1, no_of_pseudo_bags)

                instance_logits_tier_2 = None

            predicted_overall = []
            predicted_overall = [self.post_process_binary(i).item() for i in decollate_batch(overall_embedding_logits)]
            
            # predicted_with_thresh = float(predicted_overall[0].item() >= 0.5)
            # print(f"Case: {p_id}")
            # print(f'Actual: {qc_dict[p_id[0]]["label"]["quality_for_fibrosis_assessment"]}, Predicted: {predicted_with_thresh}')

            predicted_logits_of_overall_test.extend(predicted_overall)
            p_ids_list.extend(p_id)
      
            # break # For debugging

        decaaf_dict_json_path = base_dir + CONFIG['decaaf_dict_json_path']

        print(f"Saving the predicted quality for fibrosis assessment to {decaaf_dict_json_path}")

        with open(decaaf_dict_json_path, 'r') as f:
            decaaf_dict = json.load(f)

        for i, p_id in enumerate(p_ids_list):
            print(f"Case: {p_id}, Predicted: {predicted_logits_of_overall_test[i]:.4f}")
            if p_id in decaaf_dict.keys():
                decaaf_dict[p_id]['predicted_quality_for_fibrosis_assessment'] = predicted_logits_of_overall_test[i]

        with open(decaaf_dict_json_path, 'w') as f:
            json.dump(decaaf_dict, f, indent=4)

