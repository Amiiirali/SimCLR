import os

import torch
import torch.nn as nn
import torchvision.models as models

####################################################
out_channel = {'alexnet': 256, 'vgg16': 512, 'vgg19': 512, 'vgg16_bn': 512, 'vgg19_bn': 512,
               'resnet18': 512, 'resnet34': 512, 'resnet50': 2048, 'resnext50_32x4d': 2048,
               'resnext101_32x8d': 2048, 'mobilenet_v2': 1280, 'mobilenet_v3_small': 576,
               'mobilenet_v3_large': 960 ,'mnasnet1_3': 1280, 'shufflenet_v2_x1_5': 1024,
               'squeezenet1_1': 512, 'efficientnet-b0': 1280, 'efficientnet-l2': 5504,
               'efficientnet-b1': 1280, 'efficientnet-b2': 1408, 'efficientnet-b3': 1536,
               'efficientnet-b4': 1792, 'efficientnet-b5': 2048, 'efficientnet-b6': 2304,
               'efficientnet-b7': 2560, 'efficientnet-b8': 2816}

feature_map = {'alexnet': -2, 'vgg16': -2,  'vgg19': -2, 'vgg16_bn': -2,  'vgg19_bn': -2,
               'resnet18': -2, 'resnet34': -2, 'resnet50': -2, 'resnext50_32x4d': -2,
               'resnext101_32x8d': -2, 'mobilenet_v2': 0, 'mobilenet_v3_large': -2,
               'mobilenet_v3_small': -2, 'mnasnet1_3': 0, 'shufflenet_v2_x1_5': -1,
               'squeezenet1_1': 0}

####################################################
class SimCLRModel(nn.Module):

    def __init__(self, base_encoder, dim=128, pretrained=False):
        super(SimCLRModel, self).__init__()

        model = getattr(models, base_encoder)
        model = model(pretrained=pretrained)

        self.feature_extract = nn.Sequential(*list(model.children())[0]) if feature_map[base_encoder]==0 \
                               else nn.Sequential(*list(model.children())[:feature_map[base_encoder]])

        pool = nn.AdaptiveAvgPool2d(1)
        layers = [pool, nn.Flatten()]
        # projection MLP
        num_ftrs = out_channel[base_encoder]
        layers  += [nn.Linear(num_ftrs, num_ftrs)]
        layers  += [nn.Linear(num_ftrs, dim)]
        self.later_embedding = nn.Sequential(*layers)


    def forward(self, x):
        h = self.feature_extract(x)
        z = self.later_embedding(h)
        return h, z

# modified from https://github.com/stanfordmlgroup/MoCo-CXR/blob/main/moco_pretraining/moco/moco/builder.py
# class SimCLRModel(nn.Module):
#
#     def __init__(self, base_encoder, dim=128, mlp=True, pretrained=False):
#         super(SimCLRModel, self).__init__()
#
#         if pretrained:
#             model = getattr(models, base_encoder)
#             self.encoder = model(pretrained=True)
#             if self.encoder.__class__.__name__.lower() == 'resnet':
#                 num_ftrs = self.encoder.fc.in_features
#                 self.encoder.fc = nn.Linear(num_ftrs, dim)
#             elif self.encoder.__class__.__name__.lower() == 'vgg':
#                 num_ftrs = self.encoder.classifier._modules['6'].in_features
#                 self.encoder.classifier = nn.Linear(num_ftrs, dim)
#             elif self.encoder.__class__.__name__.lower() == 'mnasnet':
#                 num_ftrs = self.encoder.classifier[1].in_features
#                 self.encoder.classifier = nn.Linear(num_ftrs, dim)
#             elif self.encoder.__class__.__name__.lower() == 'densenet':
#                 num_ftrs = self.encoder.classifier.in_features
#                 self.encoder.classifier = nn.Linear(num_ftrs, dim)
#
#         else:
#             model = getattr(models, base_encoder)
#             self.encoder = model(num_classes=dim)
#
#         if mlp:
#             if self.encoder.__class__.__name__.lower() == 'resnet':
#                 dim_mlp = self.encoder.fc.weight.shape[1]
#                 self.encoder.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.fc)
#             elif self.encoder.__class__.__name__.lower() == 'vgg':
#                 dim_mlp = self.encoder.classifier._modules['0'].weight.shape[1]
#                 print(dim_mlp)
#                 self.encoder.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.classifier)
#             elif self.encoder.__class__.__name__.lower() == 'mnasnet':
#                 dim_mlp = self.encoder.classifier[1].weight.shape[1]
#                 self.encoder.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.classifier)
#             elif self.encoder.__class__.__name__.lower() == 'densenet':
#                 dim_mlp = self.encoder.classifier.weight.shape[1]
#                 self.encoder.classifier = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder.classifier)
#
#     def forward(self, x):
#         return self.encoder(x)

if __name__ == "__main__":
    model = SimCLRModel("vgg19", 128)
    print(model)
    # model = getattr(models, "vgg19")
    # model = model(pretrained=False)
    # print(model)
