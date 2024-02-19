import torch.nn as nn
import torch.nn.functional as F
import timm
from transformers import AutoConfig, AutoModel
from efficientnet_pytorch import EfficientNet


class EfficientNet_b2(nn.Module): # input size 260 260
    def __init__(self, num_classes):
        super(EfficientNet_b2, self).__init__()

        # Load EfficientNet model and its configuration
        config = AutoConfig.from_pretrained('google/EfficientNet-b2')
        self.eff_net = AutoModel.from_pretrained('google/EfficientNet-b2', config=config)

        # Custom layers after EfficientNet
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adaptive pooling to get a fixed size output
        self.dropout = nn.Dropout(0.3)  # Adding dropout for regularization
        self.fc = nn.Linear(self.eff_net.config.hidden_dim, num_classes)  # Custom fully connected layer

    def forward(self, x):
        # Pass input through EfficientNet
        eff_net_output = self.eff_net(x)
        
        # Extract the output of EfficientNet
        x = eff_net_output.last_hidden_state
        x = F.relu(x)  # Adding ReLU activation
        
        # Additional custom layers
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = self.dropout(x) # Applying dropout
        x = self.fc(x)

        return x