import copy
import os, csv, logging, datetime, argparse
import numpy as np
import pandas as pd
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
import random
from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from dataset.waterbirds_dataset import Waterbirds
from dataset.celeba_dataset import CelebADataset
from dataset.metashift_dataset import MetaShiftDataset
from dataset.dominoes_dataset import DominoesDataset
from dataset.multinli_dataset import MultiNLIDataset
from dataset.civilcomments_dataset import CivilCommentsDataset

from model import ResNet18, SimpleCNN, initialize_model, BertSequenceClassifier

from utils import evaluate_model, train_model, get_clean_noisy_indices
from utils import compute_prediction_changes, calculate_moving_average, compute_nearest_neighbour_distances, collect_data
from utils import load_precomputed_distances, precompute_distances
import ast
import json
from model import initialize_fixed_model

from sieve.pipeline import run_sieve_erm
from sieve.models import ResNet50 as SieveResNet50
from sieve.utils import evaluate_model as sieve_evaluate_model

from validation_selection import run_validation_selection_process, remove_validation_from_training, remove_samples_from_dataset, run_iterative_validation_selection, load_validation_indices
from selection_config import get_selection_config, apply_selection_config

now = datetime.datetime.now()


import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


DATASET_NAMES = ['waterbirds', 'celeba', 'metashift', 'dominoes', 'multinli', 'civilcomments']

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--dataset', type=str, help='dataset to use.')
    parser.add_argument('--target_name')
    parser.add_argument('--confounder_names', nargs='+')
    parser.add_argument('--fraction', type=float, default=1.0)
    parser.add_argument('--root_dir', default=None)
    parser.add_argument('--augment_data', action='store_true', default=False)
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    parser.add_argument('--model_type', type=str, default='simple_cnn')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight_decay')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--n_epochs', default=10, type=int, help='number of epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--confounder_percentage', default=0, type=float, help='confounder_percentage')

    parser.add_argument('--method', type=str, default='sieve-erm', choices=['sieve-erm'], help='Which method to run')
    parser.add_argument('--noise_type', type=str, default='spurious', choices=['spurious', 'symmetric'])
    parser.add_argument('--use_weighted_loss', action='store_true', default=False, help='Use weighted loss to handle imbalanced dataset')
    parser.add_argument('--flip_proportion', type=float, default=0.0, help='Proportion of labels to flip randomly in the training set')

    parser.add_argument('--distance_mode', type=str, default='fixed', 
                        choices=['pixel', 'training', 'fixed'],
                        help='Mode for computing nearest neighbour distances')
    parser.add_argument('--distance_model', type=str, default='resnet50',
                        choices=['resnet18', 'resnet50'],
                        help='Model to use for distance computation in fixed mode')

    parser.add_argument('--erm_epochs', type=int, default=100, help='Number of epochs for ERM training')
    parser.add_argument("--step_size", type=float, default=5, help="Step size for StepLR scheduler")
    parser.add_argument("--gamma", type=float, default=0.5, help="Gamma parameter for StepLR scheduler")
    parser.add_argument("--scheduler", type=str, default='none', choices=['none', 'StepLr'])

    parser.add_argument("--collect_data", action="store_true", help="Whether to collect training data for analysis")
    parser.add_argument('--save_epoch_checkpoints', action='store_true', default=False,
                        help='Save a full model+optimizer checkpoint every epoch into checkpoints/. Default OFF (was filling disk). Selection uses the in-memory best model + final models/ save, so OFF does not change training/selection/eval.')
    parser.add_argument("--val_mode", type=str, default='best_worst_group_acc',
                    choices=['best_worst_group_acc', 'use_our_metric', 'no_val_set'])
    parser.add_argument("--validation_indices_path", type=str, 
                    help="Path to saved validation indices")

    parser.add_argument("--use_val_data", action="store_true", help="Whether to collect training data for analysis")
    parser.add_argument('--balanced_class', action='store_true', default=False, 
                       help='Whether to use class-balanced training data for sieve-erm')
    parser.add_argument('--balanced_distance', action='store_true', default=False,
                       help='Whether to use balanced training set for distance computation')

    parser.add_argument("--existing_data_path", type=str, default="",
                   help="existing Json file for quick data selection pipeline test")

    parser.add_argument("--select_confusing_spurious_by", type=str, default="decrease",
                   choices=["increase", "decrease"],
                   help="the rule for choosing confusing_spurious")
    parser.add_argument("--select_confusing_non_spurious_by", type=str, default="increase",
                   choices=["increase", "decrease"],
                   help="the rule for choosing confusing_non_spurious")
    
    parser.add_argument("--remove_selected_train", action="store_true", help="Whether to remove selected examples from training set")
    parser.add_argument("--remove_selected_val", action="store_true",
                    help="Whether to remove selected examples from original validation set")

    parser.add_argument("--selected_examples_weight", type=float, default=1.0,
                    help="Weight for selected validation examples when used in training (1.0 means normal weight, 0.0 means remove)")
    parser.add_argument("--weight_decay_rate", type=float, default=0.2,
                    help="Rate at which validation sample weights decay per epoch")
    parser.add_argument("--weight_decay_type", type=str, default="linear",
                    choices=["linear", "exponential"],
                    help="Type of weight decay for validation samples")
    parser.add_argument("--use_val_data_for_selection", action="store_true", help="Whether to collect training data for analysis")
    parser.add_argument("--iterations", type=int, default=20,
                    help="Number of iterations for iterative selection")
    parser.add_argument("--n_confusing", type=int, default=200,
                   help="chosen numbers of confusing examples")
    parser.add_argument("--high_threshold_spurious_ratio", type=float, default=0.8,
                   help="High threshold ratio for spurious samples")
    parser.add_argument("--high_threshold_non_spurious_ratio", type=float, default=0.8,
                   help="High threshold ratio for non-spurious samples")


    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def initialize_dataset(args):
    if args.dataset == 'waterbirds':
        data = Waterbirds(args.root_dir, args.target_name, args.confounder_names, args.model_type, args.augment_data, 
                        confounder_percentage=args.confounder_percentage, 
                        noise_type=args.noise_type, 
                        flip_proportion=args.flip_proportion, 
                        balanced_class=args.balanced_class, 
                        balanced_distance=args.balanced_distance,
                        use_val_data=args.use_val_data)
    elif args.dataset == 'celeba':
        data = CelebADataset(args.root_dir, args.target_name, args.confounder_names, args.model_type, args.augment_data, 
                        confounder_percentage=args.confounder_percentage, 
                        noise_type=args.noise_type, 
                        flip_proportion=args.flip_proportion,
                        balanced_class=args.balanced_class, 
                        balanced_distance=args.balanced_distance,
                        use_val_data=args.use_val_data)
    elif args.dataset == 'metashift':
        data = MetaShiftDataset(args.root_dir, args.target_name, args.confounder_names, args.model_type, args.augment_data, 
                        confounder_percentage=args.confounder_percentage, 
                        noise_type=args.noise_type, 
                        flip_proportion=args.flip_proportion,
                        balanced_class=args.balanced_class, 
                        balanced_distance=args.balanced_distance,
                        use_val_data=args.use_val_data)
    elif args.dataset == 'dominoes':
        data = DominoesDataset(args.root_dir, args.target_name, args.confounder_names, args.model_type, args.augment_data, 
                        confounder_percentage=args.confounder_percentage, 
                        noise_type=args.noise_type, 
                        flip_proportion=args.flip_proportion,
                        balanced_class=args.balanced_class, 
                        balanced_distance=args.balanced_distance,
                        use_val_data=args.use_val_data)
    elif args.dataset == 'multinli':
        data = MultiNLIDataset(args=args, root_dir=args.root_dir, target_name=args.target_name, confounder_names=args.confounder_names,
                        model_type=args.model_type, augment_data=args.augment_data,
                        balanced_class=args.balanced_class,
                        balanced_distance=args.balanced_distance,
                        use_val_data=args.use_val_data)
    elif args.dataset == 'civilcomments':
        data = CivilCommentsDataset(args=args, root_dir=args.root_dir, target_name=args.target_name, confounder_names=args.confounder_names,
                        model_type=args.model_type, augment_data=args.augment_data,
                        balanced_class=args.balanced_class,
                        balanced_distance=args.balanced_distance,
                        use_val_data=args.use_val_data,
                        batch_size=args.batch_size)
    else:
        raise ValueError(f'Dataset {args.dataset} not recognized.')
    return data

def setup_experiment_paths(args):
    base_folder = f'results/sieve_erm/{args.iterations}_{args.n_confusing}_{args.high_threshold_spurious_ratio}_{args.high_threshold_non_spurious_ratio}/{args.dataset}_wd{args.wd}_lr{args.lr}_bs{args.batch_size}_ep{args.erm_epochs}/valforSelect{args.use_val_data_for_selection}_valforTrain{args.use_val_data}/{args.val_mode}_aug{args.augment_data}_save{args.collect_data}_bc{args.balanced_class}_bd{args.balanced_distance}_rmt{args.remove_selected_train}_{args.weight_decay_type}_wt{args.selected_examples_weight}_wdr{args.weight_decay_rate}'
    base_folder_share = f'results/sieve_erm/{args.iterations}_{args.n_confusing}_{args.high_threshold_spurious_ratio}_{args.high_threshold_non_spurious_ratio}/{args.dataset}_wd{args.wd}_lr{args.lr}_bs{args.batch_size}_ep{args.erm_epochs}/valforSelect{args.use_val_data_for_selection}_valforTrain{args.use_val_data}'

    general_folder_path = base_folder

    paths = {
        'general_folder_path': general_folder_path,
        'save_model': f'{general_folder_path}/models',
        'save_checkpoint': f'{general_folder_path}/checkpoints',
        'evaluate_results': f'{general_folder_path}/evaluate_results',
        'masks': f'{general_folder_path}/masks',
        'selection_model': f'{general_folder_path}/selection_model',
        'selection_data': f'{general_folder_path}/selection_data',
        'distances': f'{base_folder_share}/distance',
        'selected_examples': f'{base_folder_share}/selected_examples'
    }

    
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
        
    return paths

def main(): 
    args = parse_arguments()
    set_seed(args.seed)


    paths = setup_experiment_paths(args)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device 

    if args.method == 'sieve-erm':
        if args.val_mode == 'use_our_metric':
            print("start selection")
            selection_config = get_selection_config(args.dataset)
            selection_args = apply_selection_config(args, selection_config)
            selection_args.use_val_data = args.use_val_data_for_selection

            selection_args.iterations = args.iterations
            
            selection_args.n_confusing = args.n_confusing

            selection_args.high_threshold_spurious_ratio = args.high_threshold_spurious_ratio

            selection_args.high_threshold_non_spurious_ratio = args.high_threshold_non_spurious_ratio
            
            selection_data = initialize_dataset(selection_args)
            if args.dataset == 'celeba':
                selection_dataloaders = selection_data.prepare_dataset(
                    batch_size=args.batch_size, 
                    test_size=0.2, 
                    val_size=0.1, 
                    use_metadata_split=True,
                    results_path=paths['general_folder_path'],
                    seed=args.seed
                )
            else:
                selection_dataloaders = selection_data.prepare_dataset(
                batch_size=selection_args.batch_size, 
                test_size=0.2, 
                val_size=0.1, 
                use_metadata_split=True
                )
    

            if selection_args.model_type == 'bert':
                selection_model = BertSequenceClassifier(num_labels=selection_data.n_classes, use_pretrained=False).to(device)
            else:
                selection_model = SieveResNet50(num_classes=selection_data.n_classes).to(device)

            selected_indices_path = f"{paths['selected_examples']}/validation_indices_iterative_{selection_args.iterations}_seed{args.seed}.json"

            if os.path.exists(selected_indices_path):
                
                validation_data = load_validation_indices(selected_indices_path)
                
                if isinstance(validation_data, dict) and "indices" in validation_data:
                    args.validation_indices = validation_data["indices"]
                    args.validation_groups = validation_data["groups"] 
                else:
                    args.validation_indices = validation_data
            else:
                if hasattr(selection_args, 'iterative_selection') and selection_args.iterative_selection:
                    validation_data = run_iterative_validation_selection(
                        args=selection_args,
                        selection_model=selection_model, 
                        dataloaders=selection_dataloaders,
                        paths=paths,
                        device=device
                    )
                if isinstance(validation_data, dict) and "indices" in validation_data:
                    args.validation_indices = validation_data["indices"]
                    args.validation_groups = validation_data["groups"] 
                else:
                    args.validation_indices = validation_data
                    raise ValueError("Error: there is no group label for selected examples!")


        print("selection finished")
        data = initialize_dataset(args)
        if args.dataset == 'celeba':
            dataloaders = data.prepare_dataset(
                batch_size=args.batch_size, 
                test_size=0.2, 
                val_size=0.1, 
                use_metadata_split=True,
                results_path=paths['general_folder_path'],
                seed=args.seed
            )
        else:
            dataloaders = data.prepare_dataset(
                batch_size=args.batch_size, 
                test_size=0.2, 
                val_size=0.1, 
                use_metadata_split=True
            )

        if args.remove_selected_train:
            dataloaders = remove_validation_from_training(dataloaders, args.validation_indices)

        
        if args.val_mode == 'use_our_metric' and selection_args.use_val_data == True and args.use_val_data == False and args.remove_selected_val and 'val' in dataloaders:
            dataloaders['val'] = remove_samples_from_dataset(
                dataloaders['val'], args.validation_indices, 'val')

        if args.collect_data:
            fixed_model = initialize_fixed_model(args.distance_model, num_classes=2, 
                                            device=device, learning_rate=args.lr,
                                            weight_decay=args.wd)
            distances_path = f"{paths['save_model']}/precomputed_distances.pt"
            
            if os.path.exists(distances_path):
                distances = load_precomputed_distances(distances_path)
            else:
                distances = precompute_distances(dataloaders['distance'], 
                                            fixed_model, distances_path, device=device)
            
            args.distances = distances

        print("start training")
        model = run_sieve_erm(args, dataloaders, paths)
        print("training finished")

        final_test_loss, final_group_loss_test, final_accuracy_group_test, final_overall_accuracy_test = sieve_evaluate_model(
            dataloaders['test'], model, device, test=True)
        valid_final_groups = {g: acc for g, acc in final_accuracy_group_test.items() if acc is not None}
        worst_group_acc = min(valid_final_groups.values()) if valid_final_groups else 0.0
        print(f"Overall Test Accuracy: {final_overall_accuracy_test:.4f}")
        print(f"Worst Group Accuracy (4-group): {worst_group_acc:.4f}")


        with open(f"{paths['evaluate_results']}/{args.dataset}-final-sieve-erm.txt", 'a') as f:
            f.write(f"Final Test Results Seed {args.seed}:\n")
            f.write(f"Overall Test Accuracy: {final_overall_accuracy_test:.4f}\n")
            f.write(f"Worst Group Accuracy (4-group): {worst_group_acc:.4f}\n")

        if args.dataset == 'civilcomments':
            from eval_16group import eval_16group_from_model
            results_16g = eval_16group_from_model(
                model, dataloaders['test'], args.root_dir, split='test', device=device)
            with open(f"{paths['evaluate_results']}/{args.dataset}-final-sieve-erm.txt", 'a') as f:
                f.write(f"\n16-Group Evaluation (WILDS-compatible):\n")
                f.write(f"Worst Group Accuracy (16-group): {results_16g['worst_group_accuracy']:.2f}%\n")
                f.write(f"Overall Accuracy: {results_16g['overall_accuracy']:.2f}%\n")
    else:
        raise ValueError(f'Method {args.method} not recognized.')


if __name__ == "__main__":
    main()
