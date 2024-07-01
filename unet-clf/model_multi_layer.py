import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF


class DoubleConv(nn.Module):  # same as above.
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class MultiLevelUnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=5, features=[64, 128, 256, 512]):
        super(MultiLevelUnet, self).__init__()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        self.features_out = []

        # Constructing the down part of U-Net
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Classifier for multi-level features
        # Including bottleneck features
        total_features = sum(features) + features[-1] * 2
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        feature_maps = []

        # Downsample path
        for down in self.downs:
            x = down(x)
            feature_maps.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        feature_maps.append(x)

        # Concatenate all feature maps
        global_features = [F.adaptive_avg_pool2d(feature, (1, 1)).view(
            x.size(0), -1) for feature in feature_maps]
        global_features = torch.cat(global_features, 1)

        # Classification
        x = self.classifier(global_features)
        return x

# model = MultiLevelUnet(in_channels=1, num_classes=5)  # grayscale input for MRI
