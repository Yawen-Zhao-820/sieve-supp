
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, Subset
from transformers import BertTokenizer
from dataset.dataset import GeneralDataset


class CivilCommentsDataset(GeneralDataset):

    def __init__(self, args, root_dir, target_name, confounder_names, model_type,
                 augment_data, balanced_class=False, balanced_distance=False,
                 use_val_data=False, batch_size=None):

        self.root_dir = root_dir
        self.target_name = target_name
        self.confounder_names = confounder_names
        self.model_type = model_type
        self.augment_data = augment_data
        self.balanced_class = balanced_class
        self.balanced_distance = balanced_distance
        self.use_val_data = use_val_data
        self.confounder_percentage = 0.0

        self.data_dir = os.path.join(self.root_dir, 'data')
        metadata_path = os.path.join(self.data_dir, 'all_data_with_identities.csv')
        if not os.path.exists(metadata_path):
            raise ValueError(f'{metadata_path} does not exist. '
                             'Download CivilComments from https://worksheets.codalab.org/worksheets/0x517ef0d1c5e34f8cb90417832132231e')

        self.metadata_df = pd.read_csv(metadata_path, index_col=0)

        toxicity_col = self.metadata_df[target_name]
        n_nan_tox = toxicity_col.isna().sum()
        if n_nan_tox > 0:
            print(f"WARNING: {n_nan_tox} NaN values in '{target_name}', treating as non-toxic (0)")
        self.y_array = (toxicity_col.fillna(0).values >= 0.5).astype('int64')
        self.n_classes = 2

        confounder_col = self.metadata_df[confounder_names[0]]
        n_nan_conf = confounder_col.isna().sum()
        if n_nan_conf > 0:
            print(f"WARNING: {n_nan_conf} NaN values in '{confounder_names[0]}', treating as no identity (0)")
        self.confounder_array = (confounder_col.fillna(0).values >= 0.5).astype('int64')
        self.n_confounders = len(confounder_names)

        self.n_groups = self.n_classes * 2
        self.group_array = (self.y_array * 2 + self.confounder_array).astype('int64')

        split_col = self.metadata_df['split']
        if split_col.dtype == object or pd.api.types.is_string_dtype(split_col):
            split_map = {'train': 0, 'val': 1, 'test': 2}
            mapped = split_col.map(split_map)
            if mapped.isna().any():
                bad = split_col[mapped.isna()].unique()
                raise ValueError(f"Unknown split values: {bad}. Expected: train, val, test.")
            self.split_array = mapped.values.astype('int64')
        else:
            self.split_array = split_col.values.astype('int64')

        self.split_dict = {'train': 0, 'val': 1, 'test': 2}

        local_tokenizer_path = os.path.join(self.root_dir, 'bert-base-uncased-tokenizer')
        if os.path.isdir(local_tokenizer_path):
            self.tokenizer = BertTokenizer.from_pretrained(local_tokenizer_path)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        effective_bs = batch_size or getattr(args, 'batch_size', 24)
        if effective_bs >= 32:
            self.max_length = 128
        elif effective_bs >= 24:
            self.max_length = 220
        else:
            self.max_length = 300

        text_array = self.metadata_df['comment_text'].fillna('').tolist()

        encodings = self.tokenizer(
            text_array,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        self.x_array = torch.stack([
            encodings['input_ids'],
            encodings['attention_mask'],
            encodings['token_type_ids']
        ], dim=2)

        assert len(self.y_array) == len(self.x_array)

        self.filename_array = np.arange(len(self.y_array)).astype(str)
        self.label_array = np.copy(self.y_array)

        self.gradcam_abs_dict = {}
        self.gradcam_euc_dict = {}
        self.gradcam_cos_dict = {}
        self.gradcam_sum_dict = {}
        self.gradcam_value_dict = {}
        self.masked_images = {}

        for split_name, split_id in self.split_dict.items():
            mask = self.split_array == split_id
            n = mask.sum()
            group_counts = {g: int((self.group_array[mask] == g).sum()) for g in range(self.n_groups)}

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        x = self.x_array[idx]
        y = torch.tensor(self.y_array[idx], dtype=torch.long)
        c = torch.tensor(self.confounder_array[idx], dtype=torch.long)
        g = torch.tensor(self.group_array[idx], dtype=torch.long)
        label = torch.tensor(self.label_array[idx], dtype=torch.long)

        return x, y, c, g, str(idx), label, idx

    def get_masked_image(self, idx):
        return None
