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

class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # for LayerNorm (LN)
        # self.pretrained.layer1[0].bn1 = nn.LayerNorm(64)
        # self.pretrained.layer1[0].bn2 = nn.LayerNorm(64)
        # self.pretrained.layer1[1].bn1 = nn.LayerNorm(64)
        # self.pretrained.layer1[1].bn2 = nn.LayerNorm(64)
        # self.pretrained.layer2[0].bn1 = nn.LayerNorm(32)
        # self.pretrained.layer2[0].bn2 = nn.LayerNorm(32)
        # self.pretrained.layer2[1].bn1 = nn.LayerNorm(32)
        # self.pretrained.layer2[1].bn2 = nn.LayerNorm(32)
        # self.pretrained.layer3[0].bn1 = nn.LayerNorm(16)
        # self.pretrained.layer3[0].bn2 = nn.LayerNorm(16)
        # self.pretrained.layer3[1].bn1 = nn.LayerNorm(16)
        # self.pretrained.layer3[1].bn2 = nn.LayerNorm(16)
        # self.pretrained.layer4[0].bn1 = nn.LayerNorm(8)
        # self.pretrained.layer4[0].bn2 = nn.LayerNorm(8)
        # self.pretrained.layer4[1].bn1 = nn.LayerNorm(8)
        # self.pretrained.layer4[1].bn2 = nn.LayerNorm(8)

        # for InstanceNorm (IN)
        # self.pretrained.layer1[0].bn1 = nn.InstanceNorm2d(64)
        # self.pretrained.layer1[0].bn2 = nn.InstanceNorm2d(64)
        # self.pretrained.layer1[1].bn1 = nn.InstanceNorm2d(64)
        # self.pretrained.layer1[1].bn2 = nn.InstanceNorm2d(64)
        # self.pretrained.layer2[0].bn1 = nn.InstanceNorm2d(32)
        # self.pretrained.layer2[0].bn2 = nn.InstanceNorm2d(32)
        # self.pretrained.layer2[1].bn1 = nn.InstanceNorm2d(32)
        # self.pretrained.layer2[1].bn2 = nn.InstanceNorm2d(32)
        # self.pretrained.layer3[0].bn1 = nn.InstanceNorm2d(16)
        # self.pretrained.layer3[0].bn2 = nn.InstanceNorm2d(16)
        # self.pretrained.layer3[1].bn1 = nn.InstanceNorm2d(16)
        # self.pretrained.layer3[1].bn2 = nn.InstanceNorm2d(16)
        # self.pretrained.layer4[0].bn1 = nn.InstanceNorm2d(8)
        # self.pretrained.layer4[0].bn2 = nn.InstanceNorm2d(8)
        # self.pretrained.layer4[1].bn1 = nn.InstanceNorm2d(8)
        # self.pretrained.layer4[1].bn2 = nn.InstanceNorm2d(8)

        self.pretrained.fc = nn.Linear(512, 256)
        self.classifier1 = nn.Linear(256, 128)
        self.classifier2 = nn.Linear(128, 2)

        # For GradCAM
        self.target = [self.pretrained.layer4[-1]]
    
    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        features = self.pretrained(x)
        flattened_features = torch.max(features, 0, keepdim=True)[0]
        out = self.classifier1(flattened_features)
        out = self.classifier2(out)
        return out

class ShufflenetV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'shufflenet_v2_x1_0', weights=models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.classifier1 = nn.Linear(in_features = 1000, out_features = 256, bias = True)
        self.classifier2 = nn.Linear(in_features=256, out_features=2)

        self.target = [self.pretrained.conv5[-1]]

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.pretrained(x)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

_model_entrypoints = {
    "mrnet": MRNet,
    "resnet50": Resnet50,
    "resnet18": Resnet18,
    "shufflenetv2": ShufflenetV2,
}

def create_model(model, **kargs):
    if model in _model_entrypoints:
        model_constructor = _model_entrypoints[model]
        model = model_constructor(**kargs)
        return model
    else:
        raise RuntimeError(f"Unknown model ({model})")
