
import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertTokenizer

IDENTITY_COLS = ['male', 'female', 'LGBTQ', 'christian', 'muslim', 'other_religions', 'black', 'white']


def build_16group_labels(metadata_df, split_mask):
    sub_df = metadata_df[split_mask].copy()
    y = (sub_df['toxicity'].fillna(0).values >= 0.5).astype(int)

    group_names = []
    group_membership = np.zeros((len(sub_df), 16), dtype=bool)

    for i, col in enumerate(IDENTITY_COLS):
        identity_mask = (sub_df[col].fillna(0).values >= 0.5)
        group_names.append(f"{col}_toxic")
        group_membership[:, 2 * i] = identity_mask & (y == 1)
        group_names.append(f"{col}_nontoxic")
        group_membership[:, 2 * i + 1] = identity_mask & (y == 0)

    return y, group_membership, group_names


def compute_16group_wga(predictions, y_true, group_membership, group_names):
    predictions = np.asarray(predictions).ravel()
    y_true = np.asarray(y_true).ravel()
    assert len(predictions) == len(y_true) == group_membership.shape[0], \
        f"Shape mismatch: predictions={len(predictions)}, y_true={len(y_true)}, groups={group_membership.shape[0]}"
    correct = (predictions == y_true)
    results = {}
    group_accs = {}

    for g in range(16):
        mask = group_membership[:, g]
        count = mask.sum()
        if count > 0:
            acc = correct[mask].mean() * 100
        else:
            acc = None
        group_accs[group_names[g]] = {'accuracy': acc, 'count': int(count)}

    valid_accs = {k: v['accuracy'] for k, v in group_accs.items() if v['accuracy'] is not None}
    worst_group = min(valid_accs, key=valid_accs.get)
    wga = valid_accs[worst_group]

    overall_acc = correct.mean() * 100

    results['group_accuracies'] = group_accs
    results['worst_group'] = worst_group
    results['worst_group_accuracy'] = wga
    results['overall_accuracy'] = overall_acc
    results['n_groups_evaluated'] = len(valid_accs)

    return results


def eval_16group_from_predictions(predictions, root_dir, split='test'):
    metadata_path = os.path.join(root_dir, 'data', 'all_data_with_identities.csv')
    metadata_df = pd.read_csv(metadata_path, index_col=0)

    split_col = metadata_df['split']
    if split_col.dtype == object or pd.api.types.is_string_dtype(split_col):
        split_mask = (split_col == split).values
    else:
        split_map = {'train': 0, 'val': 1, 'test': 2}
        split_mask = (split_col == split_map[split]).values

    y_true, group_membership, group_names = build_16group_labels(metadata_df, split_mask)
    assert len(predictions) == len(y_true), f"predictions ({len(predictions)}) != split size ({len(y_true)})"

    return compute_16group_wga(predictions, y_true, group_membership, group_names)


def eval_16group_from_model(model, dataloader, root_dir, split='test', device='cuda'):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in dataloader:
            x = batch[0].to(device)
            outputs = model(x)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
    all_preds = np.concatenate(all_preds)
    return eval_16group_from_predictions(all_preds, root_dir, split)


def print_results(results):
    print(f"\n{'='*60}")
    print(f"16-Group CivilComments Evaluation (WILDS-compatible)")
    print(f"{'='*60}")
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Worst-Group Accuracy (16-group): {results['worst_group_accuracy']:.2f}%")
    print(f"Worst Group: {results['worst_group']}")
    print(f"Groups evaluated: {results['n_groups_evaluated']}/16")
    print(f"\nPer-group breakdown:")
    print(f"{'Group':<25} {'Accuracy':>10} {'Count':>8}")
    print(f"{'-'*45}")
    for name, info in results['group_accuracies'].items():
        acc_str = f"{info['accuracy']:.2f}%" if info['accuracy'] is not None else "N/A"
        print(f"{name:<25} {acc_str:>10} {info['count']:>8}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_file', type=str, help='Path to .npy file with predictions')
    parser.add_argument('--root_dir', type=str, default='dataset_metadata/civilcomments')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val'])
    args = parser.parse_args()

    if args.predictions_file:
        preds = np.load(args.predictions_file)
        results = eval_16group_from_predictions(preds, args.root_dir, args.split)
        print_results(results)
    else:
        print("Usage: python eval_16group.py --predictions_file <path.npy> --root_dir <path> --split test")
        print("Or import eval_16group_from_predictions() / eval_16group_from_model() in your code.")
