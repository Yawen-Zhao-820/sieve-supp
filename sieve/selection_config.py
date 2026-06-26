
import copy 

SELECTION_CONFIG = {
    'lr': 0.001,                
    'wd': 0.0001,            
    'epochs': 1,               
    'batch_size': 128,       
    'balanced_class': False, 
    'balanced_distance': False, 
    'collect_data': True, 
    'augment_data': False,
    'select_confusing_spurious_by': 'decrease',
    'select_confusing_non_spurious_by': 'increase',
    'iterative_selection': True,
}

DATASET_SPECIFIC_CONFIG = {
    'celeba': {
        'lr': 0.001,
        'wd': 0.0001,
        'batch_size': 128,
        'iterative_selection': True,
        'balanced_class': True, 
        'balanced_distance': True, 
    },
    'dominoes': {
        'lr': 0.001,
        'wd': 0.001,
        'batch_size': 128,
        'iterative_selection': True,
    },
    'waterbirds': {
        'lr': 0.001,
        'wd': 0.001,
        'batch_size': 128,
        'iterative_selection': True,
    },
    'metashift': {
        'lr': 0.001,
        'wd': 1e-3,
        'batch_size': 16,
        'iterative_selection': True,
    },
    'multinli': {
        'lr': 2e-5,
        'wd': 0.0,
        'epochs': 1,
        'batch_size': 32,
        'augment_data': False,
        'collect_data': True,
        'balanced_class': False,
        'balanced_distance': False,
        'iterative_selection': True,
    },
    'civilcomments': {
        'lr': 2e-5,
        'wd': 0.0,
        'epochs': 1,
        'batch_size': 24,
        'augment_data': False,
        'collect_data': True,
        'balanced_class': False,
        'balanced_distance': False,
        'iterative_selection': True,
    }
}


def get_selection_config(dataset):
    config = SELECTION_CONFIG.copy()
    
    if dataset in DATASET_SPECIFIC_CONFIG:
        config.update(DATASET_SPECIFIC_CONFIG[dataset])
    
    return config

REQUIRED_PARAMS = [
    'lr', 'wd', 'epochs', 'batch_size', 'collect_data', 'augment_data', 
    'balanced_class', 'balanced_distance'
]


def apply_selection_config(args, selection_config):
    args_copy = copy.deepcopy(args)
    
    missing_params = [param for param in REQUIRED_PARAMS if param not in selection_config]
    if missing_params:
        raise AttributeError(f"Lacking the necessary configs: {', '.join(missing_params)}")
    
    for key, value in selection_config.items():
        setattr(args_copy, key, value)
    
    return args_copy
