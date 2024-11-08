from torch import nn
from models import NRP, resnet, preact_resnet
import torch

class perceptual_criteria(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.target_layer = nn.Sequential(*list(model.children())[:-1]).eval()
        self.mse = nn.MSELoss()

    def forward(self, poi, ori):
        out = self.mse(self.target_layer(poi), self.target_layer(ori))
        return out
