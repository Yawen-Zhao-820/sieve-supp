
import torch
import torch.nn as nn
from tqdm import tqdm

class ERMTrainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.train_criterion = nn.CrossEntropyLoss(reduction='none')
        self.analysis_criterion = nn.CrossEntropyLoss(reduction='none')

    def train_epoch(self, dataloader, args, current_epoch, mode='train'):
        self.model.train()
        running_losses = {}
        batch_losses = {}
        predictions_dict = {}
        epoch_loss = 0
        count = 0

        if mode == 'train':
            if args.weight_decay_type == 'linear' and args.weight_decay_rate > 0:
                current_weight = max(0, args.selected_examples_weight * (1 - args.weight_decay_rate * current_epoch))
            elif args.weight_decay_type == 'exponential' and args.weight_decay_rate > 0:
                current_weight = args.selected_examples_weight * (args.weight_decay_rate ** current_epoch)
            else:
                current_weight = args.selected_examples_weight
            
            sample_weights = None
            if hasattr(args, 'validation_indices') and args.validation_indices and not args.remove_selected_train:
                if current_weight < 1.0:
                    sample_weights = {}
                    for idx in args.validation_indices:
                        sample_weights[idx] = current_weight
        else:
            sample_weights = None

        for batch_idx, (inputs, labels, _, _, _, mix_labels, indices) in enumerate(tqdm(dataloader)):
            inputs = inputs.to(self.device)
            labels = mix_labels.to(self.device)  

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            per_sample_losses = self.train_criterion(outputs, labels)
            if sample_weights is not None:
                batch_weights = torch.ones_like(per_sample_losses)
                for i, idx in enumerate(indices):
                    idx = idx.item()
                    if idx in sample_weights:
                        batch_weights[i] = sample_weights[idx]
                per_sample_losses = per_sample_losses * batch_weights

            loss = per_sample_losses.mean()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            count += 1

            if args.collect_data:
                self.model.eval()
                with torch.no_grad():
                    sample_losses = self.analysis_criterion(outputs, mix_labels.to(self.device))
                    _, predictions = torch.max(outputs, 1)
                    for idx, (sample_loss, pred) in zip(indices, zip(sample_losses, predictions)):
                        idx = idx.item()
                        if idx not in running_losses:
                            running_losses[idx] = []
                        running_losses[idx].append(sample_loss.item())

                        if idx not in batch_losses:
                            batch_losses[idx] = []
                        batch_losses[idx].append((batch_idx, sample_loss.item()))

                        if idx not in predictions_dict:
                            predictions_dict[idx] = []
                        predictions_dict[idx].append(pred.item())
                self.model.train()


        if self.scheduler is not None:
            self.scheduler.step()

        return epoch_loss / count, running_losses, batch_losses, predictions_dict
