import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import AlexNet_Weights

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

_model_entrypoints = {
    "mrnet": MRNet,
    "resnet50": Resnet50
}

def create_model(model, **kargs):
    if model in _model_entrypoints:
        model_constructor = _model_entrypoints[model]
        model = model_constructor(**kargs)
        return model
    else:
        raise RuntimeError(f"Unknown model ({model})")
