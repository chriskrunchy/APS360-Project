import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MultiLevelUnet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, features=[64, 128, 256, 512],
                 width_coefficient=1.0, depth_coefficient=1.0, resolution_coefficient=1.0, dropout_rate=0.5):
        super(MultiLevelUnet, self).__init__()
        
        # Apply width scaling
        features = [math.ceil(f * width_coefficient) for f in features]
        
        # Apply depth scaling (repeat blocks)
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        # Downsample path
        for i, feature in enumerate(features):
            num_repeats = math.ceil((i + 1) * depth_coefficient)
            for _ in range(num_repeats):
                self.downs.append(DoubleConv(in_channels, feature))
                in_channels = feature

        # Bottleneck
        bottleneck_features = math.ceil(features[-1] * 2 * width_coefficient)
        self.bottleneck = DoubleConv(features[-1], bottleneck_features)

        # Classifier for multi-level features
        # Including bottleneck features
        total_features = sum(features) + bottleneck_features
        self.shared_fc = nn.Sequential(
            nn.Linear(total_features, math.ceil(1024 * width_coefficient)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(math.ceil(1024 * width_coefficient), math.ceil(512 * width_coefficient)),
            nn.ReLU(inplace=True),
            nn.Linear(math.ceil(512 * width_coefficient), num_classes)
        )
        
        # Apply resolution scaling (modify input resolution)
        self.resolution_coefficient = resolution_coefficient

    def forward(self, x):
        # Adjust input resolution
        x = F.interpolate(x, scale_factor=self.resolution_coefficient, mode='bilinear', align_corners=False)
        
        feature_maps = []

        # Downsample path
        for down in self.downs:
            x = down(x)
            feature_maps.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        feature_maps.append(x)

        # Concatenate all feature maps
        global_features = [F.adaptive_avg_pool2d(feature, (1, 1)).view(x.size(0), -1) for feature in feature_maps]
        global_features = torch.cat(global_features, 1)

        # Shared fully connected layer
        x = self.shared_fc(global_features)

        return x

# Example of initializing with scaling factors
model = MultiLevelUnet(width_coefficient=1.2, depth_coefficient=1.2, resolution_coefficient=1.1, dropout_rate=0.3)
