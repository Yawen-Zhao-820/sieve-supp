

import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        d = self.model.fc.in_features
        self.model.fc = nn.Linear(d, num_classes)

    def forward(self, x):
        return self.model(x)

    def get_target_layer(self):
        return self.model.layer4[-1]
