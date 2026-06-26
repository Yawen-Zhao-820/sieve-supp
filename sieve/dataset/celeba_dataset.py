
import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from dataset.dataset import GeneralDataset

model_attributes = {
    'resnet18': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': True
    },
    'resnet50': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': True
    },
    'simple_cnn': {
        'feature_type': 'image',
        'target_resolution': (224, 224),
        'flatten': False
    },
}


class CelebADataset(GeneralDataset):

    def __init__(self, root_dir, target_name, confounder_names,
                 model_type, augment_data, confounder_percentage, noise_type, flip_proportion=0.0,
                 balanced_class=False, balanced_distance=False, use_val_data=False):
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.augment_data = augment_data
        self.model_type = model_type
        self.confounder_percentage = confounder_percentage
        self.use_val_data = use_val_data

        self.attrs_df = pd.read_csv(
            os.path.join(root_dir, 'public_datasets/CelebA', 'celebA', 'list_attr_celeba.csv'))

        self.data_dir = os.path.join(self.root_dir, 'public_datasets/CelebA', 'celebA', 'img_align_celeba')
        self.filename_array = self.attrs_df['image_id'].values
        self.attrs_df = self.attrs_df.drop(labels='image_id', axis='columns')
        self.attr_names = self.attrs_df.columns.copy()

        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0

        target_idx = self.attr_idx(self.target_name)
        self.y_array = self.attrs_df[:, target_idx]
        self.n_classes = 2

        self.confounder_idx = [self.attr_idx(a) for a in self.confounder_names]
        self.n_confounders = len(self.confounder_idx)
        confounders = self.attrs_df[:, self.confounder_idx]
        confounder_id = confounders @ np.power(2, np.arange(len(self.confounder_idx)))
        self.confounder_array = confounder_id

        self.n_groups = self.n_classes * pow(2, len(self.confounder_idx))
        self.group_array = (self.y_array*(self.n_groups/2) + self.confounder_array).astype('int')

        
        self.split_df = pd.read_csv(
            os.path.join(root_dir, 'public_datasets/CelebA', 'celebA', 'list_eval_partition.csv'))
        self.split_array = self.split_df['partition'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        if model_attributes[self.model_type]['feature_type']=='precomputed':
            self.features_mat = torch.from_numpy(np.load(
                os.path.join(root_dir, 'features', model_attributes[self.model_type]['feature_filename']))).float()
            self.train_transform = None
            self.eval_transform = None
        else:
            self.features_mat = None
            self.train_transform = get_transform_celebA(self.model_type, train=True, augment_data=augment_data)
            self.eval_transform = get_transform_celebA(self.model_type, train=False, augment_data=augment_data)

        self._split_labels(noise_type=noise_type, flip_proportion=flip_proportion) 
        self.gradcam_abs_dict = {}
        self.gradcam_euc_dict = {}
        self.gradcam_cos_dict = {}
        self.gradcam_sum_dict = {}
        self.gradcam_value_dict = {}


        self.masked_images = {}
        
        self.predictions = {}

        self.balanced_class = balanced_class
        self.balanced_distance = balanced_distance
        

    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)
    

    def update_masked_image(self, idx, masked_image):
            self.masked_images[idx] = masked_image

    def get_masked_image(self, idx):
            return self.masked_images.get(idx, None)

    def _load_balanced_partition(self, results_path=None, seed=None):
        run_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        load_path = os.path.join(run_dir, 'metadata', 'celeba_metadata.csv')
        df = pd.read_csv(load_path)
        assert df['image_id'].is_unique
        assert set(df['image_id']) == set(self.filename_array)
        assert set(df['partition']).issubset({-1, 0, 1, 2})
        filename_to_idx = {fn: i for i, fn in enumerate(self.filename_array)}
        for _, row in df.iterrows():
            self.split_array[filename_to_idx[row['image_id']]] = row['partition']
        return True

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
                    if group in [0, 3]:
                        self.label_array[group_indices[:num_confounder_samples]] = 1 - self.y_array[group_indices[:num_confounder_samples]]
                elif noise_type == 'symmetric':
                    self.label_array[group_indices[:num_confounder_samples]] = 1 - self.y_array[group_indices[:num_confounder_samples]]
                else:
                    raise ValueError(f"Noise type {noise_type} not recognized")

        if flip_proportion > 0:
            num_flip = int(num_train_samples * flip_proportion)
            flip_indices = np.random.choice(train_indices, num_flip, replace=False)
            self.label_array[flip_indices] = 1 - self.label_array[flip_indices]

def get_transform_celebA(model_type, train, augment_data):
    orig_w = 178
    orig_h = 218
    orig_min_dim = min(orig_w, orig_h)
    if model_attributes[model_type]['target_resolution'] is not None:
        target_resolution = model_attributes[model_type]['target_resolution']
    else:
        target_resolution = (orig_w, orig_h)

    if (not train) or (not augment_data):
        transform = transforms.Compose([
            transforms.CenterCrop(orig_min_dim),
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(
                target_resolution,
                scale=(0.7, 1.0),
                ratio=(1.0, 1.3333333333333333),
                interpolation=2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform
