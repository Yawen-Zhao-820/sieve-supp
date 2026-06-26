

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def _unwrap_dataset(dataset):
    base = dataset
    while hasattr(base, 'dataset'):
        base = base.dataset
    return base

def get_scheduler(optimizer, args):
    if args.scheduler == 'none':
        return None
    elif args.scheduler == 'StepLr':
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=args.step_size, 
            gamma=args.gamma
        )
    else:
        raise ValueError(f"Scheduler {args.scheduler} not supported")

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0.0)


def compute_loss_quantiles(dataset, model, quantile):
    model.eval()
    all_losses = []
    criterion = nn.CrossEntropyLoss(reduction='none')
    device = next(model.parameters()).device

    for inputs, labels, _, _, _, mix_labels, _ in tqdm(dataset):
        inputs = inputs.to(device)
        labels = mix_labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            all_losses.append(loss.detach().cpu())

    all_losses = torch.cat(all_losses).numpy()
    return np.quantile(all_losses, quantile)


def cal_sparsity(z):
    return torch.sum(torch.sum(torch.sum(z, dim=-1), -1))/(torch.numel(z[0]))


def evaluate_model(dataloader, model, device, test=False):
    model.eval()
    group_losses = defaultdict(list)
    group_correct = defaultdict(int)
    group_total = defaultdict(int)
    total_loss = 0.0
    total = 0
    base_dataset = _unwrap_dataset(dataloader.dataset)
    known_groups = getattr(base_dataset, 'n_groups', None)

    criterion_per_sample = nn.CrossEntropyLoss(reduction='none')
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels, confounders, groups, _, mix_labels, _ in dataloader:
            inputs = inputs.to(device)
            groups = groups.to(device)
            targets = labels.to(device) if test else mix_labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            per_sample_losses = criterion_per_sample(outputs, targets)
            preds = outputs.argmax(1)
            
            if known_groups is None and groups.numel() > 0:
                known_groups = int(groups.max().item()) + 1
            
            for idx_in_batch, group_idx in enumerate(groups):
                g = int(group_idx.item())
                group_losses[g].append(per_sample_losses[idx_in_batch].item())
                group_correct[g] += (preds[idx_in_batch] == targets[idx_in_batch]).item()
                group_total[g] += 1
            
            total_loss += loss.item() * inputs.size(0)
            total += inputs.size(0)

    avg_loss = total_loss / total if total > 0 else 0.0

    all_group_ids = set(group_total.keys())
    if known_groups is not None:
        all_group_ids.update(range(known_groups))

    group_accs = {
        g: (round(group_correct[g] / group_total[g] * 100.0, 4) if group_total[g] > 0 else None)
        for g in sorted(all_group_ids)
    }
    overall_acc = round(sum(group_correct.values()) / total * 100.0, 4) if total > 0 else 0.0
    group_losses = {g: group_losses[g] for g in sorted(all_group_ids)}

    return avg_loss, group_losses, group_accs, overall_acc


def evaluate_class_accuracies(model, dataloader, device):
    model.eval()
    class_correct = {}
    class_total = {}
    
    with torch.no_grad():
        for inputs, _, _, _, _, mix_labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = mix_labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            for label, pred in zip(labels, predicted):
                label_item = label.item()
                if label_item not in class_correct:
                    class_correct[label_item] = 0
                    class_total[label_item] = 0
                
                if label_item == pred.item():
                    class_correct[label_item] += 1
                class_total[label_item] += 1
    
    class_accuracies = {cls: class_correct[cls] / class_total[cls] 
                      if class_total[cls] > 0 else 0 
                      for cls in class_total}
    
    return class_accuracies
