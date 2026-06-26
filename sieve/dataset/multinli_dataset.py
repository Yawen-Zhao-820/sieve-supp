

import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
from dataset.dataset import GeneralDataset


class MultiNLIDataset(GeneralDataset):

    def __init__(self, args, root_dir, target_name, confounder_names, model_type, 
                 augment_data, balanced_class=False, balanced_distance=False, use_val_data=False):
        
        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.balanced_class = balanced_class
        self.balanced_distance = balanced_distance
        self.use_val_data = use_val_data
        self.confounder_percentage = 0.0


        self.data_dir = os.path.join(
            self.root_dir,
            'data')
        self.glue_dir = os.path.join(
            self.root_dir,
            'glue_data',
            'MNLI')
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f'{self.data_dir} does not exist yet. Please generate the dataset first.')
        if not os.path.exists(self.glue_dir):
            raise ValueError(
                f'{self.glue_dir} does not exist yet. Please generate the dataset first.')

        type_of_split = target_name.split('_')[-1]
        self.metadata_df = pd.read_csv(
            os.path.join(
                self.data_dir,
                f'metadata_{type_of_split}.csv'),
            index_col=0)

        self.y_array = self.metadata_df['gold_label'].values
        self.n_classes = len(np.unique(self.y_array))

        self.confounder_array = self.metadata_df[confounder_names[0]].values
        self.n_confounders = len(confounder_names)

        self.n_groups = len(np.unique(self.confounder_array)) * self.n_classes
        self.group_array = (self.y_array*(self.n_groups/self.n_classes) + self.confounder_array).astype('int')

        self.split_array = self.metadata_df['split'].values
        self.split_dict = {
            'train': 0,
            'val': 1,
            'test': 2
        }

        self.features_array = []
        for feature_file in [
            'cached_train_bert-base-uncased_128_mnli',  
            'cached_dev_bert-base-uncased_128_mnli',
            'cached_dev_bert-base-uncased_128_mnli-mm'
            ]:

            features = torch.load(
                os.path.join(
                    self.glue_dir,
                    feature_file))

            self.features_array += features

        self.all_input_ids = torch.tensor([f.input_ids for f in self.features_array], dtype=torch.long)
        self.all_input_masks = torch.tensor([f.input_mask for f in self.features_array], dtype=torch.long)
        self.all_segment_ids = torch.tensor([f.segment_ids for f in self.features_array], dtype=torch.long)
        self.all_label_ids = torch.tensor([f.label_id for f in self.features_array], dtype=torch.long)

        self.x_array = torch.stack((
            self.all_input_ids,
            self.all_input_masks,
            self.all_segment_ids), dim=2)

        assert np.all(np.array(self.all_label_ids) == self.y_array)

        self.filename_array = np.arange(len(self.y_array)).astype(str)
        self.label_array = np.copy(self.y_array)

        self.gradcam_abs_dict = {}
        self.gradcam_euc_dict = {}
        self.gradcam_cos_dict = {}
        self.gradcam_sum_dict = {}
        self.gradcam_value_dict = {}
        self.masked_images = {}

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        
        x = self.x_array[idx]
        y = torch.tensor(self.y_array[idx], dtype=torch.long)
        c = torch.tensor(self.confounder_array[idx], dtype=torch.long)
        g = torch.tensor(self.group_array[idx], dtype=torch.long)
        label = torch.tensor(self.y_array[idx], dtype=torch.long)

        return x, y, c, g, str(idx), label, idx

    def get_masked_image(self, idx):
        return None
