
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


class ResNetBase(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ResNetBase, self).__init__()
        self.model = base_model
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)


    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)


        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.fc(x)
        return x


    def get_representation(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x

class ResNet18(ResNetBase):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__(models.resnet18(pretrained=True), num_classes)

class ResNet50(ResNetBase):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__(models.resnet50(pretrained=True), num_classes)


class BertSequenceClassifier(nn.Module):
    def __init__(self, num_labels=3, pretrained_model_name='bert-base-uncased', use_pretrained=False, pretrained_local_dir=None):
        super().__init__()
        try:
            from transformers import BertForSequenceClassification, BertConfig
        except ImportError as exc:
            raise ImportError("Please install the `transformers` package to use BertSequenceClassifier.") from exc

        loaded = False
        if use_pretrained and pretrained_local_dir:
            config_path = os.path.join(pretrained_local_dir, 'config.json')
            weight_path = os.path.join(pretrained_local_dir, 'pytorch_model.bin')
            if os.path.exists(config_path) and os.path.exists(weight_path):
                config = BertConfig.from_json_file(config_path)
                config.num_labels = num_labels
                self.bert = BertForSequenceClassification(config)
                state_dict = torch.load(weight_path, map_location='cpu')
                remapped = {}
                for k, v in state_dict.items():
                    remapped[k if k.startswith('bert.') else f'bert.{k}'] = v
                self.bert.load_state_dict(remapped, strict=False)
                loaded = True

        if not loaded and use_pretrained:
            self.bert = BertForSequenceClassification.from_pretrained(
                pretrained_model_name,
                num_labels=num_labels,
                local_files_only=False
            )
            loaded = True

        if not loaded:
            config = BertConfig(num_labels=num_labels)
            self.bert = BertForSequenceClassification(config)

    def forward(self, x):
        input_ids = x[:, :, 0].long()
        attention_mask = x[:, :, 1].long()
        token_type_ids = x[:, :, 2].long()
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        return outputs.logits

    def get_representation(self, x):
        input_ids = x[:, :, 0].long()
        attention_mask = x[:, :, 1].long()
        token_type_ids = x[:, :, 2].long()
        outputs = self.bert.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=False
        )
        return outputs[1]


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        
        self.activations = x
        
        if x.requires_grad:
            x.register_hook(self.activations_hook)
            
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def get_activations_gradient(self):
        return self.gradients
    
    def get_activations(self, x):
        return self.activations

    def get_representation(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        return x


def initialize_model(model_type, num_classes, learning_rate, weight_decay):
    if model_type == 'resnet18':
        model = ResNet18(num_classes=num_classes)
    elif model_type == 'resnet50':
        model = ResNet50(num_classes=num_classes)
    elif model_type == 'simple_cnn':
        model = SimpleCNN(num_classes=num_classes)
    elif model_type == 'bert':
        model = BertSequenceClassifier(num_labels=num_classes)
        criterion = nn.CrossEntropyLoss(reduction='none')
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = None
        return model, criterion, optimizer, scheduler
    else:
        raise ValueError(f"Model type {model_type} not recognized.")
    
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return model, criterion, optimizer, scheduler

def initialize_fixed_model(model_type, num_classes, device, learning_rate, weight_decay):
    if model_type == 'resnet50':
        base_model = models.resnet50(pretrained=True)
        fixed_model = ResNetBase(base_model, num_classes)
    elif model_type == 'resnet18':
        base_model = models.resnet18(pretrained=True)
        fixed_model = ResNetBase(base_model, num_classes)
    else:
        raise ValueError(f"Unsupported distance model: {model_type}")

    fixed_model = fixed_model.to(device)
    fixed_model.eval()
    for param in fixed_model.parameters():
        param.requires_grad = False
    return fixed_model
