import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights
import timm
import sys

class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.classifer = nn.Linear(256, 2)

        # For GradCAM
        self.target = [self.pretrained_model.features[-1]]

    def forward(self, x):
        x = torch.squeeze(x, dim=0) 
        features = self.pretrained_model.features(x)
        pooled_features = self.pooling_layer(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        flattened_features = torch.max(pooled_features, 0, keepdim=True)[0]
        output = self.classifer(flattened_features)
        return output

class Resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.classifier = nn.Linear(1000, 2)

        # For GradCAM
        self.target = [self.pretrained.layer4[-1]]
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained(x)
        features = torch.unsqueeze(features, dim=2)
        features = torch.unsqueeze(features, dim=3)
        features = features.view(features.size(0), -1)
        flattened_features = torch.max(features, 0, keepdim=True)[0]
        out = self.classifier(flattened_features)
        return out

class SwinTiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = timm.create_model('swinv2_tiny_window16_256.ms_in1k', pretrained=True)
        self.pretrained.head.fc = nn.Linear(768, 256)
        self.classifier = nn.Sequential(nn.Linear(256, 128), nn.Linear(128, 2))

        # For GradCAM
        self.target = [self.pretrained.layers[-1].blocks[-1].norm2]
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.pretrained(x)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

_model_entrypoints = {
    "mrnet": MRNet,
    "resnet50": Resnet50,
    "swintiny": SwinTiny,
}

def create_model(model, **kargs):
    if model in _model_entrypoints:
        model_constructor = _model_entrypoints[model]
        model = model_constructor(**kargs)
        return model
    else:
        raise RuntimeError(f"Unknown model ({model})")
