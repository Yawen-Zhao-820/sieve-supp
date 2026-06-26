from .models import ResNet50 as SieveResNet50
from .erm_trainer import ERMTrainer
from .utils import get_scheduler, evaluate_model, compute_loss_quantiles, weight_init
from .utils import evaluate_class_accuracies
import torch
import torch.nn as nn
import torch.optim as optim
import copy
import os
from tqdm import tqdm
from data_collector import DataCollector
from validation_selection import evaluate_on_indices_with_custom_groups
from utils import save_checkpoint
from model import BertSequenceClassifier


def run_sieve_erm(args, dataloaders, paths):
    device = args.device

    def _unwrap_dataset(dataset):
        base = dataset
        while hasattr(base, 'dataset'):
            base = base.dataset
        return base

    base_dataset = _unwrap_dataset(dataloaders['train'].dataset)
    n_classes = getattr(base_dataset, 'n_classes', 2)

    if args.model_type == 'bert':
        local_dir = os.path.join(os.path.dirname(__file__), '..', 'pretrained')
        local_dir = os.path.abspath(local_dir)
        model = BertSequenceClassifier(num_labels=n_classes, use_pretrained=True, pretrained_local_dir=local_dir).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.wd
        )
        scheduler = None
    else:
        model = SieveResNet50(num_classes=n_classes).to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.wd
        )
        scheduler = get_scheduler(optimizer, args)    
    train_criterion = nn.CrossEntropyLoss()
    analysis_criterion = nn.CrossEntropyLoss(reduction='none')

    trainer = ERMTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    if args.collect_data:
        data_collector = DataCollector(
        dataset=dataloaders['train'].dataset,
        distances=args.distances if hasattr(args, 'distances') else None
        )
        data_collector.collect_epoch_data(0, model, dataloaders['train'], analysis_criterion, device)


    best_worst_acc = 0
    best_overall_acc = 0
    best_model = None

    best_worst_class_acc = 0
    
    for epoch in range(args.erm_epochs):
        group_accs = None
        class_accs = None
        flag_value = 0

        epoch_loss, running_losses, batch_losses, predictions = trainer.train_epoch(dataloaders['train'], args, epoch)
        if args.use_val_data == False and 'val' in dataloaders:
            val_loss, avg_group_loss_val, accuracy_group_val, overall_accuracy_val = evaluate_model(
                dataloaders['val'], model, device, test=True)
        test_loss, avg_group_loss_test, accuracy_group_test, overall_accuracy_test = evaluate_model(
                dataloaders['test'], model, device, test=True)

        worst_16group_test = None
        accuracy_16group_test = None
        worst_16group_val = None
        accuracy_16group_val = None
        if args.dataset == 'civilcomments':
            from eval_16group import eval_16group_from_model
            results_16g_test = eval_16group_from_model(
                model, dataloaders['test'], args.root_dir, split='test', device=device
            )
            accuracy_16group_test = {
                name: round(info['accuracy'], 2) if info['accuracy'] is not None else None
                for name, info in results_16g_test['group_accuracies'].items()
            }
            worst_16group_test = results_16g_test['worst_group_accuracy']
            worst_16group_test_name = results_16g_test['worst_group']
            if args.use_val_data == False and 'val' in dataloaders:
                results_16g_val = eval_16group_from_model(
                    model, dataloaders['val'], args.root_dir, split='val', device=device
                )
                accuracy_16group_val = {
                    name: round(info['accuracy'], 2) if info['accuracy'] is not None else None
                    for name, info in results_16g_val['group_accuracies'].items()
                }
                worst_16group_val = results_16g_val['worst_group_accuracy']
                worst_16group_val_name = results_16g_val['worst_group']
        
        if args.val_mode == 'best_worst_group_acc':
            if args.use_val_data == False and 'val' in dataloaders:
                valid_groups = {g: acc for g, acc in accuracy_group_val.items() if acc is not None}
                group_accs = accuracy_group_val
                class_accs = {}

                if valid_groups:
                    worst_group_acc = min(valid_groups.values())
                else:
                    worst_group_acc = 0

                if worst_group_acc > best_worst_acc:
                    best_worst_acc = worst_group_acc
                    best_model = copy.deepcopy(model)
                    best_overall_acc = overall_accuracy_val
                elif worst_group_acc == best_worst_acc and overall_accuracy_val > best_overall_acc:
                    best_model = copy.deepcopy(model)
                    best_overall_acc = overall_accuracy_val
        elif args.val_mode == 'use_our_metric':
            if hasattr(args, 'validation_groups') and args.validation_indices:

                current_acc, group_accs = evaluate_on_indices_with_custom_groups(
                    model, args.validation_indices, args.validation_groups, dataloaders, device
                )
                valid_groups = {g: acc for g, acc in group_accs.items() if acc is not None}
                if valid_groups:  
                    worst_group_acc = min(valid_groups.values())
                else:
                    worst_group_acc = 0

                if args.use_val_data:
                    class_accs = evaluate_class_accuracies(model, dataloaders['train'], device)
                    train_loss, train_avg_group_loss, train_accuracy_group, train_overall_accuracy = evaluate_model(
                        dataloaders['train'], model, device, test=False)
                    
                    worst_class_acc = min(class_accs.values()) if class_accs else 0
                    flag_value = 0
                    
                    if worst_group_acc > best_worst_acc:
                        best_worst_acc = worst_group_acc
                        best_worst_class_acc = worst_class_acc
                        best_overall_acc = train_overall_accuracy 
                        best_model = copy.deepcopy(model)
                        flag_value = 1
                    
                    elif worst_group_acc == best_worst_acc and worst_class_acc > best_worst_class_acc:
                        best_worst_class_acc = worst_class_acc
                        best_overall_acc = train_overall_accuracy  
                        best_model = copy.deepcopy(model)
                        flag_value = 2
                    
                    elif worst_group_acc == best_worst_acc and worst_class_acc == best_worst_class_acc and train_overall_accuracy > best_overall_acc:
                        best_model = copy.deepcopy(model)
                        best_overall_acc = train_overall_accuracy
                        flag_value = 3
                    
                    else:
                        flag_value = 0
                
                else:
                    class_accs = evaluate_class_accuracies(model, dataloaders['val'], device)
                    worst_class_acc = min(class_accs.values()) if class_accs else 0

                    if worst_group_acc > best_worst_acc:
                        best_worst_acc = worst_group_acc

                        best_worst_class_acc = worst_class_acc
                        best_overall_acc = current_acc

                        best_model = copy.deepcopy(model)
                    elif worst_group_acc == best_worst_acc and worst_class_acc > best_worst_class_acc:
                        best_worst_class_acc = worst_class_acc

                        best_overall_acc = current_acc
                        
                        best_model = copy.deepcopy(model)
                    elif worst_group_acc == best_worst_acc and worst_class_acc == best_worst_class_acc and  overall_accuracy_val > best_overall_acc:
                        best_model = copy.deepcopy(model)
                        best_overall_acc = overall_accuracy_val
            else:
                raise ValueError("Error: there is no group label for selected examples!")   
        
        
        elif args.val_mode == 'no_val_set':
            best_model = model
            group_accs = {}
            class_accs = {}

        if args.collect_data:
            data_collector.update_training_data(epoch+1, running_losses, batch_losses, predictions)
            data_collector.collect_epoch_data(epoch+1, model, dataloaders['train'], analysis_criterion, device)
        
        if args.dataset in ('civilcomments', 'multinli') or getattr(args, 'save_epoch_checkpoints', False):
            save_checkpoint(model, optimizer, scheduler, epoch, paths, best_overall_acc)
            
        with open(f"{paths['evaluate_results']}/training_logbook_{args.dataset}-{args.seed}.txt", 'a') as f:
            f.write(f'ERM Epoch {epoch+1}:\n')
            f.write(f'Training Loss: {epoch_loss:.4f}\n')
            if args.use_val_data == False and 'val' in dataloaders:
                val_4g_wga = min((v for v in accuracy_group_val.values() if v is not None), default=0)
                f.write(f'Val Overall Accuracy: {overall_accuracy_val:.4f}\n')
                f.write(f'Val 4-Group Accuracies: {accuracy_group_val}\n')
                f.write(f'Val 4-Group WGA: {val_4g_wga:.4f}\n')
            if worst_16group_val is not None:
                f.write(f'Val 16-Group WGA: {worst_16group_val:.2f}%\n')
                f.write(f'Val 16-Group Accuracies: {accuracy_16group_val}\n')
            test_4g_wga = min((v for v in accuracy_group_test.values() if v is not None), default=0)
            f.write(f'Test Overall Accuracy: {overall_accuracy_test:.4f}\n')
            f.write(f'Test 4-Group Accuracies: {accuracy_group_test}\n')
            f.write(f'Test 4-Group WGA: {test_4g_wga:.4f}\n')
            if worst_16group_test is not None:
                f.write(f'Test 16-Group WGA: {worst_16group_test:.2f}%\n')
                f.write(f'Test 16-Group Accuracies: {accuracy_16group_test}\n')
            f.write(f'Best Worst-Group Accuracy So Far: {best_worst_acc:.4f}\n')
            f.write(f'Best Overall Accuracy So Far: {best_overall_acc:.4f}\n')
            if group_accs is not None:
                f.write(f'Selection Val Group Accuracies: {group_accs}\n')
            f.write(f'\n')

    model.load_state_dict(best_model.state_dict())
    model.eval()
    model_save_path = f"{paths['save_model']}/model_seed{args.seed}.pt"
    torch.save(model.state_dict(), model_save_path)
    
    if args.collect_data:
        data_collector.save(f"{paths['save_model']}/collected_data_complete_{args.seed}.json")
    
    return model
