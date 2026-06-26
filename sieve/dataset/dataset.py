

import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit


class GeneralDataset(Dataset):
    def __init__(self, root_dir, target_name, confounder_names, model_type=None, augment_data=None, confounder_percentage=0.0):
        raise NotImplementedError
    
    
    def __len__(self):
        return len(self.filename_array)
    
    def __getitem__(self, idx):
        y = self.y_array[idx]
        c = self.confounder_array[idx]
        g = self.group_array[idx]
        filename = self.filename_array[idx]
        label = self.label_array[idx]

        img_filename = os.path.join(self.data_dir, self.filename_array[idx])
        img = Image.open(img_filename).convert('RGB')

        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
                img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and self.eval_transform):
                img = self.eval_transform(img)

        masked_image = self.get_masked_image(idx)
        if masked_image is not None:
            x = masked_image

        x = img
        return x, y, c, g, filename, label, idx
    

    def update_gradcam_diff(self, idx, gradcam_diff, mode):
        if mode == 'abs':
            self.gradcam_abs_dict[idx] = gradcam_diff
        elif mode == 'euclidean':
            self.gradcam_euc_dict[idx] = gradcam_diff
        elif mode == 'cosine':
            self.gradcam_cos_dict[idx] = gradcam_diff
        elif mode == 'sum':
            self.gradcam_sum_dict[idx] = gradcam_diff
        elif mode == 'value':
            self.gradcam_value_dict[idx] = gradcam_diff


    def get_gradcam_dict(self, mode):
        if mode == 'abs':
            return self.gradcam_abs_dict
        elif mode == 'euclidean':
            return self.gradcam_euc_dict
        elif mode == 'cosine':
            return self.gradcam_cos_dict
        elif mode == 'sum':
            return self.gradcam_sum_dict
        elif mode == 'value':
            return self.gradcam_value_dict
        

    def _split_labels(self, noise_type='spurious', flip_proportion=0.0):
        num_samples = len(self.y_array)
        self.label_array = np.copy(self.y_array)

        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        num_train_samples = len(train_indices)
        
        for group in range(self.n_groups):
            group_indices = np.where((self.group_array == group) & (self.split_array == self.split_dict['train']))[0]
            num_group_samples = len(group_indices)
            num_confounder_samples = int(num_group_samples * self.confounder_percentage)

            if num_confounder_samples > 0:
                np.random.shuffle(group_indices)
                if noise_type == 'spurious':
                    self.label_array[group_indices[:num_confounder_samples]] = self.confounder_array[group_indices[:num_confounder_samples]]
                elif noise_type == 'symmetric':
                    self.label_array[group_indices[:num_confounder_samples]] = 1 - self.y_array[group_indices[:num_confounder_samples]]
                else:
                    raise ValueError(f"Noise type {noise_type} not recognized")

        if flip_proportion > 0:
            num_flip = int(num_train_samples * flip_proportion)
            flip_indices = np.random.choice(train_indices, num_flip, replace=False)
            self.label_array[flip_indices] = 1 - self.label_array[flip_indices]


    def get_splits(self, splits, train_frac = 1.0):
        subsets = {}

        for split in splits:
            assert split in self.split_dict, f"Split: {split} not recognized."
            split_indices = self.split_array == self.split_dict[split]
            num_split = np.sum(split_indices)
            indices = torch.arange(len(self.split_array))[split_indices]
            split_dataset = Subset(self, indices)
            subsets[split] = split_dataset

            if train_frac < 1 and split == 'train':
                num_retain = int(np.round(float(num_split) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_retain])

            subsets[split] = Subset(self, indices)

        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)
        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names): 
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name
        

    def prepare_dataset(self, batch_size, test_size=0.2, val_size=0.1, use_metadata_split=True, save_balanced_partition=False, results_path=None, seed=None, load_existing_partition=True):
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        val_indices = np.where(self.split_array == self.split_dict['val'])[0]
        test_indices = np.where(self.split_array == self.split_dict['test'])[0]
        
        original_train_indices = train_indices.copy()

        existing_partition_loaded = False
        if load_existing_partition and seed is not None:
            existing_partition_loaded = self._load_balanced_partition(results_path, seed)
            
            if existing_partition_loaded:
                train_indices = np.where(self.split_array == self.split_dict['train'])[0]
                val_indices = np.where(self.split_array == self.split_dict['val'])[0]
                test_indices = np.where(self.split_array == self.split_dict['test'])[0]

        if not existing_partition_loaded:
            if self.use_val_data:
                combined_mask = (self.split_array == self.split_dict['train']) | (self.split_array == self.split_dict['val'])
                self.split_array[combined_mask] = self.split_dict['train']
                train_indices = np.where(self.split_array == self.split_dict['train'])[0]
                val_indices = np.array([])

            if self.balanced_class:
                train_labels = self.y_array[train_indices]
                unique_labels = np.unique(train_labels)
                class_indices = {label: train_indices[train_labels == label] for label in unique_labels}
                
                class_sizes = {label: len(indices) for label, indices in class_indices.items()}
                
                if len(set(class_sizes.values())) == 1:
                    balanced_train_indices = train_indices
                else:
                    min_class_size = min(class_sizes.values())
                    balanced_train_indices = []
                    
                    for label, indices in class_indices.items():
                        if len(indices) == min_class_size:
                            balanced_train_indices.extend(indices)
                        else:
                            balanced_train_indices.extend(
                                np.random.choice(indices, min_class_size, replace=False)
                            )
                    
                    balanced_train_indices = np.array(balanced_train_indices)
                    train_indices = balanced_train_indices
                    
                    new_split_array = self.split_array.copy()
                    train_mask = np.zeros_like(new_split_array, dtype=bool)
                    for idx in train_indices:
                        train_mask[idx] = True
                    
                    removed_mask = np.zeros_like(new_split_array, dtype=bool)
                    pre_balance_train_indices = np.where(self.split_array == self.split_dict['train'])[0]
                    for idx in pre_balance_train_indices:
                        if idx not in train_indices:
                            removed_mask[idx] = True
                    
                    new_split_array[removed_mask] = -1
                    self.split_array = new_split_array

                    self._save_balanced_partition(results_path, seed=seed)
            
            elif save_balanced_partition:
                self._save_balanced_partition(results_path, seed=seed)
        
        if self.balanced_distance:
            distance_train_indices = train_indices
        else:
            distance_train_indices = original_train_indices

        train_set = Subset(self, train_indices)
        val_set = Subset(self, val_indices)
        test_set = Subset(self, test_indices)
        distance_train_set = Subset(self, distance_train_indices)

        dataloaders = {
            'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=6),
            'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=6),
            'test': DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=6),
            'test-train': DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=6),
            'distance': DataLoader(distance_train_set, batch_size=batch_size, shuffle=False, num_workers=6)
        }

        return dataloaders

    def _load_balanced_partition(self, results_path=None, seed=None):
        if not hasattr(self, 'filename_array') or seed is None:
            return False
        import pandas as pd
        import os
        if results_path:
            save_dir = results_path
        elif hasattr(self, 'data_dir'):
            save_dir = os.path.dirname(self.data_dir)
        else:
            save_dir = self.root_dir
        load_path = os.path.join(save_dir, f'list_eval_partition_balanced_seed{seed}.csv')
        if os.path.exists(load_path):
            try:
                df = pd.read_csv(load_path)
                filename_to_idx = {filename: idx for idx, filename in enumerate(self.filename_array)}
                for _, row in df.iterrows():
                    image_id = row['image_id']
                    partition = row['partition']
                    if image_id in filename_to_idx:
                        idx = filename_to_idx[image_id]
                        self.split_array[idx] = partition
                return True
            except Exception as e:
                return False
        else:
            return False

    
    def get_num_examples(self):
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        val_indices = np.where(self.split_array == self.split_dict['val'])[0]
        test_indices = np.where(self.split_array == self.split_dict['test'])[0]
        return train_indices, val_indices, test_indices


    def get_num_examples_per_group(self, indices):
        num_examples_per_group = np.bincount(self.group_array[indices])
        return num_examples_per_group   


    def _save_balanced_partition(self, results_path=None, seed=None):
        if hasattr(self, 'split_array') and hasattr(self, 'filename_array'):
            import pandas as pd
            import os
            
            data = {
                'image_id': self.filename_array,
                'partition': self.split_array
            }
            df = pd.DataFrame(data)
            
            if results_path:
                save_dir = results_path
            elif hasattr(self, 'data_dir'):
                save_dir = os.path.dirname(self.data_dir)
            else:
                save_dir = self.root_dir
            
            save_path = os.path.join(save_dir, f'list_eval_partition_balanced_seed{seed}.csv')
            df.to_csv(save_path, index=False)
