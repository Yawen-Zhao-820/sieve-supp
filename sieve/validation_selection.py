
import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from collections import Counter

from sieve.erm_trainer import ERMTrainer
import gzip
from data_collector import DataCollector
import random
from model import initialize_fixed_model, BertSequenceClassifier
from utils import precompute_distances,load_precomputed_distances, precompute_cls_prototype_distances
from sieve.models import ResNet50 as SieveResNet50


def load_sample_data(file_path):
    opener = gzip.open if file_path.endswith('.gz') else open
    with opener(file_path, 'rt') as f:
        return json.load(f)


def assign_group_by_spuriousness(dataset_name, ground_truth, is_spurious):
    ground_truth = int(ground_truth)  
    
    if dataset_name == 'multinli':
        if ground_truth == 0:
            return 1 if is_spurious else 0
        elif ground_truth == 1:
            return 2 if is_spurious else 3
        elif ground_truth == 2:
            return 4 if is_spurious else 5
        else:
            raise ValueError(f"Unexpected MultiNLI label: {ground_truth}")

    if dataset_name in ['waterbirds', 'dominoes', 'civilcomments']:
        if is_spurious:
            group = 0 if ground_truth == 0 else 3
        else:
            group = 1 if ground_truth == 0 else 2
    elif dataset_name in ['celeba', 'isic']:
        if is_spurious:
            group = 1 if ground_truth == 0 else 2
        else:
            group = 0 if ground_truth == 0 else 3
    elif dataset_name == 'metashift':
        if ground_truth == 0:
            group = 0 if is_spurious else 1
        else:
            group = 2 if is_spurious else 3
    else:
        if is_spurious:
            group = ground_truth * 2 + 1
        else:
            group = ground_truth * 2
    
    return group


def select_validation_samples(data, dataset_name, 
                             running_loss=False, 
                             high_threshold_spurious_ratio=0.8, 
                             high_threshold_non_spurious_ratio=0.8,
                             low_threshold_spurious_ratio=0.1,
                             low_threshold_non_spurious_ratio=0.1,
                             start_epoch=0, end_epoch=1, 
                             use_final_loss=False,
                             n_confusing=1596,
                             select_confusing_spurious_by="decrease",
                             select_confusing_non_spurious_by="increase"):

    from analyze_loss_patterns import classify_confusing_samples
    

    result, accuracy = classify_confusing_samples(
        data=data,
        running_loss=running_loss,
        high_threshold_spurious_ratio=high_threshold_spurious_ratio,
        high_threshold_non_spurious_ratio=high_threshold_non_spurious_ratio,
        low_threshold_spurious_ratio=low_threshold_spurious_ratio,
        low_threshold_non_spurious_ratio=low_threshold_non_spurious_ratio,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        use_final_loss=use_final_loss,
        n_confusing=n_confusing,
        dataset=dataset_name,
        select_confusing_spurious_by=select_confusing_spurious_by,
        select_confusing_non_spurious_by=select_confusing_non_spurious_by
    )
    
    validation_indices = set()
    validation_indices.update(result["confusing_spurious"])
    validation_indices.update(result["confusing_non_spurious"])
    

    validation_groups = {}
    
    for idx in result["confusing_spurious"]:
        ground_truth = data['ground_truths'][str(idx)]
        group = assign_group_by_spuriousness(dataset_name, ground_truth, True)
        validation_groups[idx] = group
    
    for idx in result["confusing_non_spurious"]:
        ground_truth = data['ground_truths'][str(idx)]
        group = assign_group_by_spuriousness(dataset_name, ground_truth, False)
        validation_groups[idx] = group
    
    return {
        "all_indices": validation_indices,
        "confusing_spurious": set(result["confusing_spurious"]),
        "confusing_non_spurious": set(result["confusing_non_spurious"]),
        "accuracy": accuracy,
        "groups": validation_groups
    }

def run_validation_selection_process(args, model, dataloaders, paths, device):
    if hasattr(args, 'validation_indices_path') and args.validation_indices_path and os.path.exists(args.validation_indices_path):
        validation_data = load_validation_indices(args.validation_indices_path)
        if isinstance(validation_data, dict):
            if "all_indices" in validation_data:
                indices = validation_data["all_indices"]
                groups = validation_data.get("groups", {})
            elif "indices" in validation_data:
                indices = validation_data["indices"]
                groups = validation_data.get("groups", {})
            else:
                indices = validation_data
                groups = {}
        else:
            indices = validation_data
            groups = {}
        
        return {"indices": indices, "groups": groups}
    
    if hasattr(args, 'existing_data_path') and args.existing_data_path and os.path.exists(args.existing_data_path):
        try:
            with open(args.existing_data_path, 'r') as f:
                collected_data = json.load(f)
        except Exception as e:
            raise
    else:
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd
        )
        
        scheduler = None
        
        trainer = ERMTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        

        data_collector = DataCollector(
            dataset=dataloaders['train'].dataset,
            distances=args.distances if hasattr(args, 'distances') else None
        )
        
        data_collector.collect_epoch_data(0, model, dataloaders['train'], trainer.analysis_criterion, device)
        
        for epoch in range(1, args.epochs + 1):
            
            epoch_loss, running_losses, batch_losses, predictions = trainer.train_epoch(dataloaders['train'], args)
            
            data_collector.update_training_data(epoch, running_losses, batch_losses, predictions)
            data_collector.collect_epoch_data(epoch, model, dataloaders['train'], trainer.analysis_criterion, device)
        
        
        data_path = f"{paths['selection_data']}/collected_data_bs{args.batch_size}_aug{args.augment_data}_val{args.use_val_data}_bc{args.balanced_class}_bd{args.balanced_distance}_seed{args.seed}.json.gz"
        data_collector.save(data_path)

    collected_data = load_sample_data(data_path)
        
    validation_indices_result = select_validation_samples(
        data=collected_data,
        dataset_name=args.dataset,
        running_loss=False,
        start_epoch=0,
        end_epoch=args.epochs,
        n_confusing=args.n_confusing if hasattr(args, 'n_confusing') else 100,
        select_confusing_spurious_by=args.select_confusing_spurious_by if hasattr(args, 'select_confusing_spurious_by') else "decrease",
        select_confusing_non_spurious_by=args.select_confusing_non_spurious_by if hasattr(args, 'select_confusing_non_spurious_by') else "increase",
        high_threshold_spurious_ratio=args.high_threshold_spurious_ratio,
        high_threshold_non_spurious_ratio=args.high_threshold_non_spurious_ratio
    )
    
    validation_data = {
        "indices": validation_indices_result["all_indices"],
        "groups": validation_indices_result["groups"],
        "confusing_spurious": validation_indices_result["confusing_spurious"],
        "confusing_non_spurious": validation_indices_result["confusing_non_spurious"],
        "accuracy": validation_indices_result["accuracy"]
    }
    
    if not hasattr(args, 'no_save_validation') or not args.no_save_validation:
        save_path = f"{paths['selection_data']}/validation_indices_bs{args.batch_size}_aug{args.augment_data}_val{args.use_val_data}_bc{args.balanced_class}_bd{args.balanced_distance}_seed{args.seed}.json"
        save_validation_indices(validation_data, {
            "method": "loss_pattern_analysis",
            "dataset": args.dataset,
            "epochs": [0, args.epochs],
            "seed": args.seed,
            "params": {
                "running_loss": False,
                "n_confusing": args.n_confusing if hasattr(args, 'n_confusing') else 100,
                "select_confusing_spurious_by": args.select_confusing_spurious_by if hasattr(args, 'select_confusing_spurious_by') else "decrease",
                "select_confusing_non_spurious_by": args.select_confusing_non_spurious_by if hasattr(args, 'select_confusing_non_spurious_by') else "increase"
            }
        }, save_path)
    
    return validation_data
   

def save_validation_indices(validation_data, metadata, save_path):
    serializable_data = {}
    
    if isinstance(validation_data, dict):
        if "indices" in validation_data:
            serializable_data["validation_indices"] = list(validation_data["indices"])
            if "groups" in validation_data:
                serializable_data["group_labels"] = {str(k): int(v) for k, v in validation_data["groups"].items()}
            if "confusing_spurious" in validation_data:
                serializable_data["confusing_spurious"] = list(validation_data["confusing_spurious"])
            if "confusing_non_spurious" in validation_data:
                serializable_data["confusing_non_spurious"] = list(validation_data["confusing_non_spurious"])
            if "accuracy" in validation_data:
                serializable_data["accuracy"] = validation_data["accuracy"]
        elif "all_indices" in validation_data:
            serializable_data["validation_indices"] = list(validation_data["all_indices"])
            if "groups" in validation_data:
                serializable_data["group_labels"] = {str(k): int(v) for k, v in validation_data["groups"].items()}
    else:
        serializable_data["validation_indices"] = list(validation_data)
    
    serializable_data["selection_criteria"] = metadata
    
    with open(save_path, 'w') as f:
        json.dump(serializable_data, f)

def load_validation_indices(load_path):
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    result = {}
    
    if "validation_indices" in data:
        result["indices"] = set(data["validation_indices"])
        
        if "group_labels" in data:
            result["groups"] = {int(k): v for k, v in data["group_labels"].items()}
            
        for key in ["confusing_spurious", "confusing_non_spurious", "accuracy"]:
            if key in data:
                result[key] = set(data[key]) if isinstance(data[key], list) else data[key]
    else:
        result = set(data)
    
    return result


def evaluate_on_indices_with_custom_groups(model, indices, groups, dataloaders, device):
    model.eval()
    group_ids = set(groups.values()) if groups else set()
    if not group_ids:
        group_ids = {0, 1, 2, 3, 4, 5}
    else:
        max_gid = max(group_ids)
        group_ids.update(range(max_gid + 1))
    correct_by_group = {g: 0 for g in group_ids}  
    total_by_group = {g: 0 for g in group_ids}
    total_correct = 0
    total_samples = 0
    
    indices_set = set(indices)
    
    if 'custom_val' in dataloaders:
        with torch.no_grad():
            for inputs, _, _, _, _, mix_labels, batch_indices in dataloaders['custom_val']:
                inputs = inputs.to(device)
                labels = mix_labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                for i, idx in enumerate(batch_indices):
                    idx_item = idx.item()
                    if idx_item in groups:
                        group = groups[idx_item]
                        is_correct = (predicted[i] == labels[i]).item()
                        correct_by_group[group] += is_correct
                        total_by_group[group] += 1
                        total_correct += is_correct
                        total_samples += 1
    else:
        with torch.no_grad():
            for inputs, _, _, _, _, mix_labels, batch_indices in dataloaders['train']:
                indices_in_batch = []
                mask = []
                batch_groups = []
                
                for i, idx in enumerate(batch_indices):
                    idx_item = idx.item()
                    if idx_item in indices_set and idx_item in groups:
                        indices_in_batch.append(i)
                        mask.append(True)
                        batch_groups.append(groups[idx_item])
                    else:
                        mask.append(False)
                
                if not indices_in_batch:
                    continue
                    
                mask = torch.tensor(mask)
                inputs = inputs[mask].to(device)
                labels = mix_labels[mask].to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                for i, group in enumerate(batch_groups):
                    is_correct = (predicted[i] == labels[i]).item()
                    correct_by_group[group] += is_correct
                    total_by_group[group] += 1
                    total_correct += is_correct
                    total_samples += 1

    group_accuracies = {g: (correct_by_group[g] / total_by_group[g] if total_by_group[g] > 0 else None) 
                    for g in total_by_group}
    overall_accuracy = total_correct / total_samples if total_samples > 0 else None
    

    return overall_accuracy, group_accuracies


from torch.utils.data import Dataset, Subset
from torch.utils.data import DataLoader
import copy

    
def remove_samples_from_dataset(dataloader, indices_to_remove, dataset_name='train'):
    if not indices_to_remove:
        return dataloader

    indices_to_remove_set = set(indices_to_remove)
    original_dataset = dataloader.dataset.dataset
    dataset_indicdes = dataloader.dataset.indices

    global_to_local = {global_idx: local_idx for local_idx, global_idx in enumerate(dataset_indicdes)}
    dataset_indicdes_filtered = [idx for idx in dataset_indicdes if idx not in indices_to_remove_set]
    new_dataset = Subset(original_dataset, dataset_indicdes_filtered)

    if dataset_name == 'train':
        shuffle = True
    else:
        shuffle = False

    new_dataloader = DataLoader(
        new_dataset,
        batch_size=dataloader.batch_size,
        shuffle=shuffle,
        num_workers=getattr(dataloader, 'num_workers', 0),
        pin_memory=getattr(dataloader, 'pin_memory', False)
    )
    
    removed_count = len(dataset_indicdes) - len(dataset_indicdes_filtered)
    
    return new_dataloader
    

def remove_validation_from_training(dataloaders, validation_indices):

    updated_loaders = dataloaders.copy()
    updated_loaders['original_train'] = dataloaders['train']
    
    validation_indices_set = set(validation_indices)

    original_train_dataset = dataloaders['train'].dataset.dataset
    train_indices = dataloaders['train'].dataset.indices
    
    train_indices_filtered = [idx for idx in train_indices if idx not in validation_indices_set]
    new_train_dataset = Subset(original_train_dataset, train_indices_filtered)

    updated_loaders['train'] = DataLoader(
        new_train_dataset,
        batch_size=dataloaders['train'].batch_size,
        shuffle=True,
        num_workers=getattr(dataloaders['train'], 'num_workers', 0),
        pin_memory=getattr(dataloaders['train'], 'pin_memory', False)
    )

    if 'distance' in dataloaders:
        distance_dataset = dataloaders['distance'].dataset
        if isinstance(distance_dataset, Subset):
            distance_indices = distance_dataset.indices
            distance_indices_filtered = [idx for idx in distance_indices if idx not in validation_indices_set]
            new_distance_dataset = Subset(distance_dataset.dataset, distance_indices_filtered)
            
            updated_loaders['distance'] = DataLoader(
                new_distance_dataset,
                batch_size=dataloaders['distance'].batch_size,
                shuffle=False, 
                num_workers=getattr(dataloaders['distance'], 'num_workers', 0),
                pin_memory=getattr(dataloaders['distance'], 'pin_memory', False)
            )
    
    custom_val_datasets = []
    
    train_val_global_indices = [idx for idx in validation_indices if idx in train_indices]
    if train_val_global_indices:
        train_val_dataset = Subset(original_train_dataset, train_val_global_indices)
        custom_val_datasets.append(train_val_dataset)
    
    if 'val' in dataloaders and len(dataloaders['val'].dataset) > 0:
        val_dataset = dataloaders['val'].dataset.dataset
        val_indices = dataloaders['val'].dataset.indices
        
        val_val_global_indices = [idx for idx in validation_indices if idx in val_indices]
        if val_val_global_indices:
            val_val_dataset = Subset(val_dataset, val_val_global_indices)
            custom_val_datasets.append(val_val_dataset)
    
    if custom_val_datasets:
        if len(custom_val_datasets) > 1:
            from torch.utils.data import ConcatDataset
            custom_val_dataset = ConcatDataset(custom_val_datasets)
        else:
            custom_val_dataset = custom_val_datasets[0]
        
        updated_loaders['custom_val'] = DataLoader(
            custom_val_dataset,
            batch_size=dataloaders['train'].batch_size,
            shuffle=False,
            num_workers=getattr(dataloaders['train'], 'num_workers', 0),
            pin_memory=getattr(dataloaders['train'], 'pin_memory', False)
        )
        
        total_samples = sum(len(dataset) for dataset in custom_val_datasets)
    return updated_loaders


def run_iterative_validation_selection(args, selection_model, dataloaders, paths, device):
    if hasattr(args, 'iterative_selection') and args.iterative_selection:
        num_iterations = args.iterations if hasattr(args, 'iterations') else 3
    else:
        num_iterations = 1
    
    
    all_validation_indices = set()
    all_validation_groups = {}
    current_dataloaders = copy.deepcopy(dataloaders)
    
    if args.model_type == 'bert' or args.dataset in ('multinli', 'civilcomments'):
        distances_path = f"{paths['distances']}/precomputed_distances_seed{args.seed}.pt"
        if os.path.exists(distances_path):
            original_distances = load_precomputed_distances(distances_path)
        else:
            selection_model = selection_model.to(device)
            original_distances = precompute_cls_prototype_distances(
                current_dataloaders['distance'],
                selection_model,
                device
            )
            torch.save({"distances": original_distances}, distances_path)
    else:
        fixed_model = initialize_fixed_model(args.distance_model, num_classes=2, 
                                           device=device, learning_rate=args.lr,
                                           weight_decay=args.wd)

        distances_path = f"{paths['distances']}/precomputed_distances_seed{args.seed}.pt"

        if os.path.exists(distances_path):
            try:
                original_distances = load_precomputed_distances(distances_path)
            except Exception as e:
                original_distances = precompute_distances(current_dataloaders['distance'], 
                                        fixed_model, distances_path, device=device)
        else:
            original_distances = precompute_distances(current_dataloaders['distance'], 
                                        fixed_model, distances_path, device=device)
    
    current_distances = copy.deepcopy(original_distances)
    
    for iteration in range(num_iterations):
        
        if args.model_type == 'bert' or args.dataset in ('multinli', 'civilcomments'):
            base_dataset = current_dataloaders['train'].dataset.dataset
            n_classes = getattr(base_dataset, 'n_classes', 3)
            local_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'pretrained'))
            model = BertSequenceClassifier(num_labels=n_classes, use_pretrained=True, pretrained_local_dir=local_dir).to(device)
            optimizer = optim.AdamW(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.wd
            )
        else:
            model = SieveResNet50().to(device)
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.wd
            )
        
        
        current_args = copy.deepcopy(args)
        current_args.distances = current_distances
        
        scheduler = None
        
        trainer = ERMTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device
        )
        
        data_collector = DataCollector(
            dataset=current_dataloaders['train'].dataset,
            distances=current_args.distances 
        )
        
        data_collector.collect_epoch_data(0, model, current_dataloaders['train'], trainer.analysis_criterion, device)
        
        for epoch in range(1, args.epochs + 1):
            
            epoch_loss, running_losses, batch_losses, predictions = trainer.train_epoch(current_dataloaders['train'], current_args, epoch, mode="selection")
            
            data_collector.update_training_data(epoch, running_losses, batch_losses, predictions)
            data_collector.collect_epoch_data(epoch, model, current_dataloaders['train'], trainer.analysis_criterion, device)
        
        data_path = f"{paths['selection_data']}/collected_data_iteration_{iteration+1}_seed{args.seed}.json.gz"
        data_collector.save(data_path)
        
        collected_data = load_sample_data(data_path)
        
        adjusted_n_confusing = args.n_confusing
        
        validation_indices_result = select_validation_samples(
            data=collected_data,
            dataset_name=args.dataset,
            running_loss=False,
            start_epoch=0,
            end_epoch=args.epochs,
            n_confusing=adjusted_n_confusing,
            select_confusing_spurious_by=args.select_confusing_spurious_by,
            select_confusing_non_spurious_by=args.select_confusing_non_spurious_by,
            high_threshold_spurious_ratio=args.high_threshold_spurious_ratio,
            high_threshold_non_spurious_ratio=args.high_threshold_non_spurious_ratio
        )
        
        selected_indices = validation_indices_result["all_indices"]
        
        all_validation_indices.update(validation_indices_result["all_indices"])
        for idx in validation_indices_result["all_indices"]:
            if idx in validation_indices_result["groups"]:
                all_validation_groups[idx] = validation_indices_result["groups"][idx]
        
        current_dataloaders = remove_validation_from_training(
            current_dataloaders, 
            validation_indices_result["all_indices"]
        )

        current_distances = {k: v for k, v in current_distances.items() if k not in selected_indices}
    
    
    validation_data = {
        "indices": all_validation_indices,
        "groups": all_validation_groups
    }
    
    save_path = f"{paths['selected_examples']}/validation_indices_iterative_{num_iterations}_seed{args.seed}.json"

    save_validation_indices(validation_data, {
        "method": "iterative_loss_pattern_analysis",
        "dataset": args.dataset,
        "iterations": num_iterations,
        "seed": args.seed
    }, save_path)
    
    
    return validation_data
