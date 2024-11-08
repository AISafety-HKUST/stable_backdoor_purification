import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseNet161(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.densenet161(num_classes=num_classes)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)

    def from_input_to_features(self, x):
        
        return self.model.features(x)
        
    def from_features_to_output(self, out):
        out = F.relu(out, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.model.classifier(out)
        return out
 