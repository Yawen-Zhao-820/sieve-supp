

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import copy
from torch.utils.data import Subset

class DominoesDataset(Dataset):
    def __init__(self, root_dir, target_name="dominoes", confounder_names=["background"], model_type="resnet50", 
                 augment_data=False, confounder_percentage=0.0, noise_type='spurious', flip_proportion=0.0,
                 balanced_class=False, balanced_distance=False, use_val_data=False):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.confounder_percentage = confounder_percentage
        self.noise_type = noise_type
        self.flip_proportion = flip_proportion

        self.n_classes = 2
        self.n_groups = 4

        self._load_data()
        self._prepare_splits()
        self._apply_noise()

        self.train_transform = self._get_transform(train=True)
        self.eval_transform = self._get_transform(train=False)


        self.balanced_class = balanced_class
        self.balanced_distance = balanced_distance
        self.use_val_data = use_val_data

    def _load_data(self):
        self.X = torch.tensor(np.load(os.path.join(self.root_dir, 'X_train.npy')))
        self.y_array = np.load(os.path.join(self.root_dir, 'y_train.npy'))
        self.group_array = np.load(os.path.join(self.root_dir, 'env_train.npy'))
        
        self.X = torch.cat([self.X, 
                            torch.tensor(np.load(os.path.join(self.root_dir, 'X_val.npy'))),
                            torch.tensor(np.load(os.path.join(self.root_dir, 'X_test.npy')))])
        self.y_array = np.concatenate([self.y_array, 
                                       np.load(os.path.join(self.root_dir, 'y_val.npy')),
                                       np.load(os.path.join(self.root_dir, 'y_test.npy'))])
        self.group_array = np.concatenate([self.group_array,
                                           np.load(os.path.join(self.root_dir, 'env_val.npy')),
                                           np.load(os.path.join(self.root_dir, 'env_test.npy'))])

        self.confounder_array = self.group_array % 2

    def _prepare_splits(self):
        train_size = np.load(os.path.join(self.root_dir, 'y_train.npy')).shape[0]
        val_size = np.load(os.path.join(self.root_dir, 'y_val.npy')).shape[0]
        
        self.split_array = np.zeros(len(self.y_array))
        self.split_array[train_size:train_size+val_size] = 1
        self.split_array[train_size+val_size:] = 2

        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

    def _apply_noise(self):
        self.label_array = np.copy(self.y_array)
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        
        self.confounder_samples_per_group = {i: 0 for i in range(self.n_groups)}

        for group in range(self.n_groups):
            group_indices = np.where((self.group_array == group) & (self.split_array == self.split_dict['train']))[0]
            num_group_samples = len(group_indices)
            num_confounder_samples = int(num_group_samples * self.confounder_percentage)

            if num_confounder_samples > 0:
                np.random.shuffle(group_indices)
                noise_indices = group_indices[:num_confounder_samples]
                
                if self.noise_type == 'spurious':
                    if group in [1, 2]:
                        self.label_array[noise_indices] = 1 - self.y_array[noise_indices]
                elif self.noise_type == 'symmetric':
                    self.label_array[noise_indices] = 1 - self.y_array[noise_indices]
                else:
                    raise ValueError(f"Noise type {self.noise_type} not recognized")
                
                self.confounder_samples_per_group[group] = num_confounder_samples

        if self.flip_proportion > 0:
            num_flip = int(len(train_indices) * self.flip_proportion)
            flip_indices = np.random.choice(train_indices, num_flip, replace=False)
            self.label_array[flip_indices] = 1 - self.label_array[flip_indices]

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            x = self.train_transform(x)
        elif self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and self.eval_transform:
            x = self.eval_transform(x)

        y = torch.tensor(self.y_array[idx], dtype=torch.long)
        c = torch.tensor(self.confounder_array[idx], dtype=torch.long)
        g = torch.tensor(self.group_array[idx], dtype=torch.long)
        label = torch.tensor(self.label_array[idx], dtype=torch.long)

        return x, y, c, g, str(idx), label, idx

    def __len__(self):
        return len(self.y_array)

    def _get_transform(self, train):
        if train and self.augment_data:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            ])
        else:
            return None

    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = torch.utils.data.Subset(self, indices)
        return subsets


    def prepare_dataset(self, batch_size, test_size=0.2, val_size=0.1, use_metadata_split=True):
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        val_indices = np.where(self.split_array == self.split_dict['val'])[0]
        test_indices = np.where(self.split_array == self.split_dict['test'])[0]

        if self.use_val_data:
            train_mask = self.split_array == self.split_dict['train']
            val_mask = self.split_array == self.split_dict['val']
            
            combined_mask = train_mask | val_mask
            self.split_array[combined_mask] = self.split_dict['train']

            train_indices = np.where(self.split_array == self.split_dict['train'])[0]
            val_indices = np.array([])

        original_train_indices = train_indices.copy()

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
