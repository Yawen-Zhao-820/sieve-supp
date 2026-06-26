import numpy as np


def get_confusing_nonconfusing_samples(sample_data, num_samples=100):
    distances = sample_data.get('distances', {})
    if distances:
        sorted_distances = sorted(distances.items(), key=lambda x: x[1])
        confusing = [int(idx) for idx, _ in sorted_distances[:num_samples]]
        nonconfusing = [int(idx) for idx, _ in sorted_distances[-num_samples:]]
    else:
        losses = sample_data.get('losses', {})
        scored = []
        for idx_str, loss_list in losses.items():
            if loss_list:
                scored.append((int(idx_str), loss_list[0]))
        scored.sort(key=lambda x: x[1], reverse=True)
        confusing = [idx for idx, _ in scored[:num_samples]]
        nonconfusing = [idx for idx, _ in scored[-num_samples:]] if scored else []
    
    return confusing, nonconfusing


def categorize_samples(sample_data, confusing, nonconfusing, dataset):
    categories = {
        'confusing': {
            'clean': {'spurious': [], 'non_spurious': []},
            'noisy': {'spurious': [], 'non_spurious': []}
        },
        'non_confusing': {
            'clean': {'spurious': [], 'non_spurious': []},
            'noisy': {'spurious': [], 'non_spurious': []}
        }
    }
    
    for idx in confusing + nonconfusing:
        str_idx = str(idx)
        gt = sample_data['ground_truths'][str_idx]
        ml = sample_data['mix_labels'][str_idx]
        c = sample_data['confounders'][str_idx]
        
        confusing_category = 'confusing' if idx in confusing else 'non_confusing'
        clean_noisy = 'clean' if gt == ml else 'noisy'
        
        sample_is_spurious = is_spurious(ml, c, dataset)
        spurious_category = 'spurious' if sample_is_spurious else 'non_spurious'
        
        categories[confusing_category][clean_noisy][spurious_category].append(idx)
    
    counts = {}
    for conf in ['confusing', 'non_confusing']:
        for clean in ['clean', 'noisy']:
            for spur in ['spurious', 'non_spurious']:
                key = (conf, clean, spur)
                counts[key] = len(categories[conf][clean][spur])
    
    return categories, counts


def is_spurious(y, c, dataset):
    if dataset in ['celeba', 'isic']:
        return (y == 0 and c == 1) or (y == 1 and c == 0)
    elif dataset in ['waterbirds', 'dominoes', 'civilcomments']:
        return (y == 0 and c == 0) or (y == 1 and c == 1)
    elif dataset == 'metashift':
        return c != 4
    elif dataset == 'multinli':
        if y == 0:
            return c == 1
        elif y in [1, 2]:
            return c == 0
        else:
            raise ValueError(f"Unexpected label for multinli: {y}")
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def classify_confusing_samples(
    data, running_loss=True, 
    high_threshold_spurious_ratio=0.8, high_threshold_non_spurious_ratio=0.8, 
    low_threshold_spurious_ratio=0.1,low_threshold_non_spurious_ratio=0.1,
    start_epoch=None, end_epoch=None, use_final_loss=True, n_confusing=500, dataset="waterbirds",
    select_confusing_spurious_by="increase",
    select_confusing_non_spurious_by="increase",
    spurious_selection_mode="speed"
):

    confusing, nonconfusing = get_confusing_nonconfusing_samples(data, num_samples=n_confusing)
    categories, counts = categorize_samples(data, confusing, nonconfusing, dataset)

    loss_key = 'running_losses' if running_loss else 'losses'
    sample_ids = list(confusing)

    increasing_samples = []
    increasing_losses = []
    decreasing_samples = []
    decreasing_losses = []

    for idx in sample_ids:
        sample_loss = data[loss_key][str(idx)]
        start = 0 if start_epoch is None else start_epoch
        end = len(sample_loss) - 1 if end_epoch is None else end_epoch
        if end >= len(sample_loss):
            continue
        loss_change = sample_loss[end] - sample_loss[start]

        if loss_change > 0:
            increasing_samples.append(idx)
            increasing_losses.append(sample_loss[end] if use_final_loss else loss_change)
        elif loss_change < 0:
            decreasing_samples.append(idx)
            if spurious_selection_mode == "speed":
                decreasing_losses.append(abs(loss_change))
            elif spurious_selection_mode == "low_value":
                decreasing_losses.append(sample_loss[end])

    if start_epoch == 0 and end_epoch == 0:
        print("Using Epoch 0 Losses for Thresholding")
        epoch0_losses = [(idx, data[loss_key][str(idx)][0]) for idx in sample_ids if str(idx) in data[loss_key] and len(data[loss_key][str(idx)]) > 0]

        all_losses = [loss for _, loss in epoch0_losses]

        if all_losses:
            high_threshold = np.percentile(all_losses, high_threshold_spurious_ratio * 100)
            low_threshold = np.percentile(all_losses, low_threshold_non_spurious_ratio * 100)

            confusing_spurious = [idx for idx, loss in epoch0_losses if loss < low_threshold]
            confusing_non_spurious = [idx for idx, loss in epoch0_losses if loss > high_threshold]
        else:
            confusing_spurious = []
            confusing_non_spurious = []
    else:
        if select_confusing_spurious_by == "increase":
            if increasing_losses:
                low_threshold_spurious = np.percentile(increasing_losses, low_threshold_spurious_ratio * 100)
            else:
                low_threshold_spurious = float('inf')

            confusing_spurious = [idx for idx, loss in zip(increasing_samples, increasing_losses) if loss < low_threshold_spurious]

        elif select_confusing_spurious_by == "decrease":
            if decreasing_losses:
                high_threshold_spurious = np.percentile(decreasing_losses, high_threshold_spurious_ratio * 100)
            else:
                high_threshold_spurious = float('inf')

            confusing_spurious = [idx for idx, loss in zip(decreasing_samples, decreasing_losses) if loss > high_threshold_spurious]

        if select_confusing_non_spurious_by == "increase":
            if increasing_losses:
                high_threshold_non_spurious = np.percentile(increasing_losses, high_threshold_non_spurious_ratio * 100)
            else:
                high_threshold_non_spurious = float('inf')
            
            confusing_non_spurious = [idx for idx, loss in zip(increasing_samples, increasing_losses) if loss > high_threshold_non_spurious]

        elif select_confusing_non_spurious_by == "decrease":
            if decreasing_losses:
                low_threshold_non_spurious = np.percentile(decreasing_losses, low_threshold_non_spurious_ratio * 100)
            else:
                low_threshold_non_spurious = float('inf')

            confusing_non_spurious = [idx for idx, loss in zip(decreasing_samples, decreasing_losses) if loss < low_threshold_non_spurious]


    prediction_stats = {
        "confusing_spurious_correct": 0,
        "confusing_spurious_wrong": 0,
        "confusing_spurious_correct_0": 0,
        "confusing_spurious_correct_1": 0,
        "confusing_spurious_misclassified": [],
        
        "confusing_non_spurious_correct": 0,
        "confusing_non_spurious_wrong": 0,
        "confusing_non_spurious_correct_0": 0,
        "confusing_non_spurious_correct_1": 0,
        "confusing_non_spurious_misclassified": []
    }

    for idx in confusing_spurious:
        gt = data['ground_truths'][str(idx)]
        confounder = data['confounders'][str(idx)]
        is_sp = is_spurious(gt, confounder, dataset)

        if is_sp:  
            prediction_stats["confusing_spurious_correct"] += 1
            if gt == 0:
                prediction_stats["confusing_spurious_correct_0"] += 1
            else:
                prediction_stats["confusing_spurious_correct_1"] += 1
        else: 
            prediction_stats["confusing_spurious_wrong"] += 1
            prediction_stats["confusing_spurious_misclassified"].append((gt, "Should be Non-Spurious"))

    for idx in confusing_non_spurious:
        gt = data['ground_truths'][str(idx)]
        confounder = data['confounders'][str(idx)]
        is_sp = is_spurious(gt, confounder, dataset)

        if not is_sp:  
            prediction_stats["confusing_non_spurious_correct"] += 1
            if gt == 0:
                prediction_stats["confusing_non_spurious_correct_0"] += 1
            else:
                prediction_stats["confusing_non_spurious_correct_1"] += 1
        else:  
            prediction_stats["confusing_non_spurious_wrong"] += 1
            prediction_stats["confusing_non_spurious_misclassified"].append((gt, "Should be Spurious"))

    correct = prediction_stats["confusing_spurious_correct"] + prediction_stats["confusing_non_spurious_correct"]
    total_confusing_samples = len(confusing_spurious) + len(confusing_non_spurious)
    accuracy = correct / total_confusing_samples if total_confusing_samples > 0 else 0

    return {
        "confusing_spurious": confusing_spurious,
        "confusing_non_spurious": confusing_non_spurious,
    }, accuracy
