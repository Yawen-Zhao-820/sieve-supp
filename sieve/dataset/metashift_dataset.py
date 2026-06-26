

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader


class MetaShiftDataset(Dataset):
    def __init__(self, root_dir, target_name="animal", confounder_names=["background"], model_type="resnet50", 
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
        self.RGB = True
        self.balanced_class = balanced_class
        self.balanced_distance = balanced_distance
        self.use_val_data = use_val_data

        self.train_data_dir = os.path.join(self.root_dir, "train")
        self.test_data_dir = os.path.join(self.root_dir, 'test')

        self.n_classes = 2
        self.n_groups = 4

        self._load_data()
        self._prepare_splits()
        self._apply_noise()

        self.train_transform = self._get_transform(train=True)
        self.eval_transform = self._get_transform(train=False)



    def _get_data(self):
        filenames = []
        ys = []
        confounders = []
        
        for background in ['sofa', 'bed']:
            path = os.path.join(self.train_data_dir, f"cat/cat({background})")
            if os.path.exists(path):
                files = os.listdir(path)
                for file in files:
                    filenames.append(os.path.join(path, file))
                    ys.append(0)
                    confounders.append(0 if background == 'sofa' else 1)
        
        for background in ['bench', 'bike']:
            path = os.path.join(self.train_data_dir, f"dog/dog({background})")
            if os.path.exists(path):
                files = os.listdir(path)
                for file in files:
                    filenames.append(os.path.join(path, file))
                    ys.append(1)
                    confounders.append(2 if background == 'bench' else 3)
        
        for animal in ['cat', 'dog']:
            background = 'shelf'
            path = os.path.join(self.test_data_dir, f"{animal}/{animal}({background})")
            if os.path.exists(path):
                files = os.listdir(path)
                for file in files:
                    filenames.append(os.path.join(path, file))
                    ys.append(0 if animal == 'cat' else 1)
                    confounders.append(4)
        
        return filenames, np.array(ys), np.array(confounders)

    def _prepare_splits(self):
        import pandas as pd
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        df = pd.read_csv(os.path.join(run_dir, 'metadata', 'metashift_metadata.csv'))
        assert df['filename'].is_unique
        assert set(df['split']).issubset({0, 1, 2})
        rel_to_split = dict(zip(df['filename'], df['split'].astype(int)))
        root = os.path.abspath(self.root_dir).rstrip('/')
        self.split_array = np.zeros(len(self.filename_array))
        for i, f in enumerate(self.filename_array):
            self.split_array[i] = rel_to_split[os.path.relpath(os.path.abspath(f), root)]
        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

    def _apply_noise(self):
        self.label_array = np.copy(self.y_array)
        train_indices = np.where(self.split_array == self.split_dict['train'])[0]
        
        self.confounder_samples_per_group = {i: 0 for i in range(self.n_groups)}

        if self.noise_type == 'symmetric':
            num_noise = int(len(train_indices) * self.flip_proportion)
            noise_indices = np.random.choice(train_indices, num_noise, replace=False)
            self.label_array[noise_indices] = 1 - self.label_array[noise_indices]
            for group in range(self.n_groups):
                group_noise = np.sum((self.group_array[noise_indices] == group))
                self.confounder_samples_per_group[group] = group_noise
        elif self.noise_type == 'spurious':
            for group in range(self.n_groups):
                group_indices = np.where((self.group_array == group) & (self.split_array == self.split_dict['train']))[0]
                num_group_samples = len(group_indices)
                num_confounder_samples = int(num_group_samples * self.confounder_percentage)

                if num_confounder_samples > 0:
                    np.random.shuffle(group_indices)
                    noise_indices = group_indices[:num_confounder_samples]
                    
                    self.label_array[noise_indices] = 1 - self.y_array[noise_indices]
                    self.confounder_samples_per_group[group] = num_confounder_samples


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

    def __getitem__(self, idx):
        img = self.get_image(idx)
        y = self.y_array[idx]
        c = self.confounder_array[idx]
        g = self.group_array[idx]
        filename = self.filename_array[idx]
        label = self.label_array[idx]
        return img, y, c, g, filename, label, idx


    def __len__(self):
        return len(self.filename_array)


    def get_image(self, idx):
        img_filename = self.filename_array[idx]
        img = Image.open(img_filename)
        if self.RGB:
            img = img.convert("RGB")
        
        if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
            img = self.train_transform(img)
        elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
            img = self.eval_transform(img)

        return img
    

    def _get_transform(self, train):
        scale = 256.0 / 224.0
        target_resolution = (224, 224)
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        if (not train) or (not self.augment_data):
            transform = transforms.Compose([
                transforms.Resize((int(target_resolution[0] * scale), int(target_resolution[1] * scale))),
                transforms.CenterCrop(target_resolution),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    target_resolution,
                    scale=(0.7, 1.0),
                    ratio=(0.75, 1.3333333333333333),
                    interpolation=2),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        return transform


    def get_splits(self, splits, train_frac=1.0):
        subsets = {}
        for split in splits:
            mask = self.split_array == self.split_dict[split]
            indices = np.where(mask)[0]
            if train_frac < 1 and split == 'train':
                num_to_retain = int(np.round(float(len(indices)) * train_frac))
                indices = np.sort(np.random.permutation(indices)[:num_to_retain])
            subsets[split] = Subset(self, indices)
        return subsets


    def _load_data(self):
        self.filename_array, self.y_array, self.confounder_array = self._get_data()
        
        self.group_array = np.zeros_like(self.y_array)
        
        for i in range(len(self.y_array)):
            y_i = self.y_array[i]
            c_i = self.confounder_array[i]
            
            if y_i == 0:
                if c_i in [0, 1]:
                    self.group_array[i] = 0
                elif c_i == 4:
                    self.group_array[i] = 1
            else:
                if c_i in [2, 3]:
                    self.group_array[i] = 2
                elif c_i == 4:
                    self.group_array[i] = 3
