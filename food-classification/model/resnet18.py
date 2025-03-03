import torch.nn as nn
import torchvision.models as models


class Resnet18FineTuneModel(nn.Module):
    def __init__(self, num_classes, dropout_rate=0.5, unfreeze_layers=("layer4", "fc")):
        super().__init__()
        
        self.base = models.resnet18(pretrained=True)
    
        for param in self.base.parameters():
            param.requires_grad = False

        for name, param in self.base.named_parameters():
            if any(layer in name for layer in unfreeze_layers):
                param.requires_grad = True  # Enable gradient updates for selected layers

        # Replace the classifier head
        self.base.fc = nn.Sequential(
            nn.BatchNorm1d(self.base.fc.in_features),
            nn.Dropout(p=dropout_rate),
            nn.Linear(self.base.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)