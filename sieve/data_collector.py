
import torch
import numpy as np
from tqdm import tqdm
import json
import gzip
from utils import collect_data

class DataCollector:
    def __init__(self, dataset, distances=None):
        selected_indices = dataset.indices
        self.data = {
            'losses': {},         
            'running_losses': {},  
            'batch_losses': {},    
            'predictions': {},     
            'distances': distances if distances is not None else {},
            'epochs': {},         
            'ground_truths': {int(idx): int(dataset.dataset.y_array[idx]) for idx in selected_indices},
            'mix_labels': {int(idx): int(dataset.dataset.label_array[idx]) for idx in selected_indices},
            'confounders': {int(idx): int(dataset.dataset.confounder_array[idx]) for idx in selected_indices},
        }
        

    def update_training_data(self, epoch, running_losses, batch_losses, predictions):
        for idx, losses in running_losses.items():
            if idx not in self.data['running_losses']:
                self.data['running_losses'][idx] = []
                self.data['batch_losses'][idx] = []
                self.data['predictions'][idx] = []
                self.data['epochs'][idx] = []

            self.data['running_losses'][idx].extend(losses)
            self.data['batch_losses'][idx].extend(batch_losses[idx])
            self.data['predictions'][idx].extend(predictions[idx])
            self.data['epochs'][idx].append(epoch)
    
    
    def collect_epoch_data(self, epoch, model, dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            epoch_losses = {}
            for inputs, _, _, _, _, mix_labels, indices in dataloader:
                inputs = inputs.to(device)
                mix_labels = mix_labels.to(device)
                outputs = model(inputs)
                losses = criterion(outputs, mix_labels)
                _, predictions = torch.max(outputs, 1)
                
                for idx, (loss, pred) in zip(indices, zip(losses, predictions)):
                    idx = idx.item()
                    if idx not in self.data['losses']:
                        self.data['losses'][idx] = []
                    if epoch == 0:
                        self.data['losses'][idx] = [loss.item()]
                        self.data['running_losses'][idx] = [loss.item()]  
                        self.data['batch_losses'][idx] = [(None, loss.item())]  
                        self.data['predictions'][idx] = [pred.item()] 
                        self.data['epochs'][idx] = [epoch]
                    else:
                        self.data['losses'][idx].append(loss.item())

    def save(self, save_path):
        keep_keys = ['losses', 'running_losses', 'distances', 'ground_truths', 'mix_labels', 'confounders']
        json_compatible_data = {}
        for key in keep_keys:
            if key in self.data:
                json_compatible_data[key] = {str(k): v for k, v in self.data[key].items()}
        
        save_path = save_path.replace('.pt', '.json.gz')
        with gzip.open(save_path, 'wt', encoding='utf-8') as f:
            json.dump(json_compatible_data, f)
