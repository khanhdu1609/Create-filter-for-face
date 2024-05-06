import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets, models, transforms
import torchvision.models as models

class ResNet18Finetune(nn.Module):
  def __init__(self, output_shape=[68, 2]):
    super().__init__()
    self.output_shape = output_shape
    backbone = models.resnet18(pretrained=True)
    layers = list(backbone.children())
    self.feature_extractor = nn.Sequential(*layers[:-1]) #Cut the fc layer in the last

    #freeze all the layers in feature extractor
    for parameter in self.feature_extractor.parameters():
      parameter.requires_grad = False

    #unfreeze some last layers:
    for param in self.feature_extractor[-2][1].parameters():
      parameter.requires_grad = True

    #get the input feature in the last layer
    num_filters = backbone.fc.in_features

    #create the fully connected layers in the last layer
    self.output_layer = nn.Linear(num_filters, self.output_shape[0]*self.output_shape[1])

  def forward(self, x):
    x = self.feature_extractor(x)

    #Flatten x
    x = x.view(x.size(0), -1)

    x = self.output_layer(x)
    x = x.view(x.size(0), self.output_shape[0], self.output_shape[1])
    return x
