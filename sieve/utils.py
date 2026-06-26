
import torch
import numpy as np
from scipy.spatial.distance import cosine
import torch.nn.functional as F
import torch.nn as nn
from torchvision import models
from collections import defaultdict

from tqdm import tqdm
import datetime
import copy
import json
import gc
import os

def _unwrap_dataset(dataset):
    base = dataset
    while hasattr(base, 'dataset'):
        base = base.dataset
    return base


def collect_data(model, dataloader, criterion, device):
    all_losses = {}
    all_predictions = {}
    
    with torch.no_grad():
        for inputs, ground_truth, confounders, groups, filename, mix_labels, indices in dataloader:
            inputs, mix_labels = inputs.to(device), mix_labels.to(device)
            outputs = model(inputs)
            losses = criterion(outputs, mix_labels)
            _, preds = torch.max(outputs, 1)
            
            for idx, loss, pred in zip(indices, losses, preds):
                idx = idx.item()
                all_losses[idx] = loss.item()
                all_predictions[idx] = pred.item()
    
    return all_losses, all_predictions


def compute_nearest_neighbour_distances(dataloader, model, device, mode='training', fixed_model=None):
    if mode not in ['pixel', 'training', 'fixed']:
        raise ValueError("Invalid mode. Choose from 'pixel', 'training', or 'fixed'")
    
    if mode == 'fixed' and model is None:
        raise ValueError("Fixed model is required for 'fixed' mode")

    all_features = []
    all_mix_labels = []
    all_indices = []

    with torch.no_grad():
        for inputs, _, _, _, _, mix_labels, indices in dataloader:
            inputs = inputs.to(device)
            if mode == 'pixel':
                features = inputs.view(inputs.size(0), -1)
            elif mode == 'training':
                features = model.get_representation(inputs)
            elif mode == 'fixed':
                if fixed_model is None:
                    raise ValueError("Fixed model is required for 'fixed' mode")
                features = fixed_model.get_representation(inputs)
            
            all_features.append(features.cpu())
            all_mix_labels.extend(mix_labels.numpy())
            all_indices.extend(indices.numpy())

    all_features = torch.cat(all_features, dim=0)
    all_mix_labels = np.array(all_mix_labels)
    all_indices = np.array(all_indices)

    nearest_neighbour_distances = {}

    for i, (feat, mix_label, idx) in enumerate(zip(all_features, all_mix_labels, all_indices)):
        diff_class_mask = all_mix_labels != mix_label
        diff_class_features = all_features[diff_class_mask]
        distances = torch.cdist(feat.unsqueeze(0), diff_class_features).squeeze()
        min_distance = distances.min().item()
        nearest_neighbour_distances[idx] = min_distance

    nearest_neighbour_distances = {int(k): v for k, v in nearest_neighbour_distances.items()}
    return nearest_neighbour_distances

def get_clean_noisy_indices(dataloader):
    clean_0_indices, noisy_0_indices = [], []
    clean_1_indices, noisy_1_indices = [], []
    for inputs, ground_truth, confounders, groups, filename, mix_labels, indices in dataloader:
        for idx, gt, ml in zip(indices, ground_truth, mix_labels):
            if ml == gt:
                if gt == 0:
                    clean_0_indices.append(idx.item())
                else:
                    clean_1_indices.append(idx.item())
            else:
                if ml == 0:
                    noisy_0_indices.append(idx.item())
                else:
                    noisy_1_indices.append(idx.item())
    return clean_0_indices, noisy_0_indices, clean_1_indices, noisy_1_indices


def train_model(dataloader, model, criterion, optimizer, device, use_weighted_loss=False):
    model.train()
    running_losses = {}
    batch_sample_losses = {}
    total_loss = 0.0
    total_samples = 0
    base_dataset = _unwrap_dataset(dataloader.dataset)
    num_classes = getattr(base_dataset, 'n_classes', None)

    if use_weighted_loss:
        inferred_classes = 0
        class_samples = torch.zeros(num_classes if num_classes is not None else 0)
        for _, ground_truth, _, _, _, mix_labels, _ in dataloader:
            inferred_classes = max(inferred_classes, int(mix_labels.max().item()) + 1)
            needed_classes = num_classes if num_classes is not None else inferred_classes
            if class_samples.numel() < needed_classes:
                class_samples = torch.zeros(needed_classes)
            class_samples += torch.bincount(mix_labels, minlength=class_samples.numel())
        
        weights = 1. / class_samples
        weights = weights / weights.sum()
        weights = weights.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=weights, reduction='none')


    for batch_idx, (inputs, ground_truth, confounders, groups, filename, mix_labels, indices) in enumerate(dataloader):
        inputs, ground_truth, confounders, groups , mix_labels, indices = \
        inputs.to(device), ground_truth.to(device), confounders.to(device), groups.to(device), mix_labels.to(device), indices.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        losses = criterion(outputs, mix_labels)
        
        loss = losses.mean()
        loss.backward() 
        optimizer.step()  

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        for idx, sample_loss in zip(indices, losses):
            idx = idx.item()
            if idx not in running_losses:
                running_losses[idx] = []
            running_losses[idx].append(sample_loss.item())

            if idx not in batch_sample_losses:
                batch_sample_losses[idx] = []
            batch_sample_losses[idx].append((batch_idx, sample_loss.item()))

    
    avg_loss = total_loss / total_samples
    avg_running_losses = {idx: sum(losses) / len(losses) for idx, losses in running_losses.items()}

    return (avg_loss, avg_running_losses, batch_sample_losses)


def evaluate_model(dataloader, model, criterion, device, test=True):
    model.eval()
    running_loss = 0.0
    total_instances = 0
    total_correct_preds = 0
    base_dataset = _unwrap_dataset(dataloader.dataset)
    known_groups = getattr(base_dataset, 'n_groups', None)

    group_loss = defaultdict(list)
    group_counts = defaultdict(int)
    correct_preds = defaultdict(int)

    with torch.no_grad():
        for inputs, ground_truth, confounders, groups , filename, mix_labels, indices in dataloader:
            inputs = inputs.to(device)
            ground_truth = ground_truth.to(device)
            groups = groups.to(device)
            mix_labels = mix_labels.to(device)

            outputs = model(inputs)
            targets = ground_truth if test else mix_labels
            losses = criterion(outputs, targets)

            if known_groups is None and groups.numel() > 0:
                known_groups = int(groups.max().item()) + 1

            preds = outputs.argmax(dim=1)
            for idx_in_batch, (loss_value, group_idx) in enumerate(zip(losses, groups)):
                group = int(group_idx.item())
                group_loss[group].append(loss_value.item())
                group_counts[group] += 1
                correct_preds[group] += (preds[idx_in_batch] == targets[idx_in_batch]).item()

            running_loss += losses.mean().item() * inputs.size(0)
            total_correct_preds += (outputs.argmax(dim=1) == targets).sum().item()
            total_instances += inputs.size(0)   

    avg_loss = running_loss / total_instances if total_instances > 0 else 0.0

    all_group_ids = set(group_counts.keys())
    if known_groups is not None:
        all_group_ids.update(range(known_groups))

    avg_group_loss = {
        group: (sum(group_loss[group]) / group_counts[group] if group_counts[group] > 0 else None)
        for group in sorted(all_group_ids)
    }
    accuracy_group = {
        group: (correct_preds[group] / group_counts[group] if group_counts[group] > 0 else None)
        for group in sorted(all_group_ids)
    }
    overall_accuracy = total_correct_preds / total_instances if total_instances > 0 else 0.0

    return avg_loss, avg_group_loss, accuracy_group, overall_accuracy
    

def compute_prediction_changes(model, dataloader, device):
    model.eval()
    predictions = []
    indices = []
    with torch.no_grad():
        for inputs, _, _, _, _, mix_labels, idx in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            indices.extend(idx.numpy())
    return predictions, indices

def calculate_moving_average(values, window_size):
    weights = np.ones(window_size) / window_size
    moving_averages = np.convolve(values, weights, mode='valid')
    return moving_averages

def precompute_cls_prototype_distances(dataloader, model, device):
    model.eval()
    all_features = []
    all_labels = []
    all_indices = []

    with torch.no_grad():
        for inputs, ground_truth, _, _, _, mix_labels, indices in tqdm(dataloader, desc="Collecting CLS embeddings"):
            inputs = inputs.to(device)
            labels = mix_labels.to(device)
            feats = model.get_representation(inputs)
            all_features.append(feats.cpu())
            all_labels.append(labels.cpu())
            all_indices.append(indices)

    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_indices = torch.cat(all_indices, dim=0)

    prototypes = {}
    for cls in torch.unique(all_labels):
        mask = all_labels == cls
        prototypes[int(cls)] = all_features[mask].mean(dim=0)

    distances = {}
    for feat, lbl, idx in zip(all_features, all_labels, all_indices):
        other_protos = [proto for c, proto in prototypes.items() if c != int(lbl)]
        if other_protos:
            stacked = torch.stack(other_protos, dim=0)
            dists = torch.norm(stacked - feat.unsqueeze(0), dim=1)
            distances[int(idx)] = dists.min().item()
        else:
            distances[int(idx)] = 0.0

    return distances


def precompute_distances(dataloader, fixed_model, save_path, device):
    distances = {}
    fixed_model.eval()
    
    num_workers = 0
    torch.set_num_threads(4)
    
    fixed_model = fixed_model.to(device)
    
    all_features = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting features"):
            inputs, _, _, _, _, mix_labels, indices = batch
            inputs = inputs.to(device)
            features = fixed_model.get_representation(inputs)
            
            all_features.append(features)
            all_labels.append(mix_labels.to(device))
            all_indices.append(indices)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_indices = torch.cat(all_indices, dim=0).cpu().numpy()
    
    
    unique_labels = torch.unique(all_labels)
    
    batch_size = 1000
    
    for label in unique_labels:
        current_mask = (all_labels == label)
        current_indices = all_indices[current_mask.cpu().numpy()]
        current_features = all_features[current_mask]
        
        diff_mask = (all_labels != label)
        diff_features = all_features[diff_mask]
        
        if len(diff_features) > 0:
            for i in range(0, len(current_indices), batch_size):
                batch_end = min(i + batch_size, len(current_indices))
                current_batch = current_features[i:batch_end]
                
                with torch.cuda.amp.autocast(enabled=True):
                    dist_matrix = torch.cdist(current_batch, diff_features)
                    min_distances = dist_matrix.min(dim=1)[0]
                
                min_distances_cpu = min_distances.cpu().numpy()
                
                for idx, min_dist in zip(current_indices[i:batch_end], min_distances_cpu):
                    distances[idx] = float(min_dist)
        else:
            for idx in current_indices:
                distances[idx] = float('inf')
    
    save_dict = {
        "dataset": os.path.basename(save_path).split('_')[0],
        "distances": distances,
        "computation_date": str(datetime.datetime.now()),
        "total_samples": len(distances)
    }
    
    torch.save(save_dict, save_path)
    
    del all_features, all_labels
    torch.cuda.empty_cache()
    
    return distances


def load_precomputed_distances(save_path):
    save_dict = torch.load(save_path)
    return save_dict["distances"]


def save_checkpoint(model, optimizer, scheduler, epoch, paths, acc, group_accs=None):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy': acc
    }
    if group_accs is not None:
        checkpoint['group_accuracies'] = group_accs
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
    torch.save(checkpoint, 
              f"{paths['save_checkpoint']}/checkpoint_epoch_{epoch+1}.pt")
    
