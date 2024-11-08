import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(num_classes=num_classes)

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        return self.model(x)

    def from_input_to_features(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        out = self.model.layer4(x)

        
        return out
        
    def from_features_to_output(self, out):
        out = self.model.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.model.fc(out)
        return out
 