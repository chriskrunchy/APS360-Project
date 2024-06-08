# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# LeNet architecture
# 1x32x32 Input -> (5x5),s=1,p=0 -> avg pool s=2,p=0 -> (5x5),s=1,p=0 -> avg pool s=2,p=0
# -> Conv 5x5 to 120 channels x Linear 84 x Linear 10


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(
            5, 5), stride=(1, 1), padding=(0, 0))
        self.s2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.s4 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # same as s2
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120,
                            kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.f6 = nn.Linear(in_features=120, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = self.s2(x)
        x = F.relu(self.c3(x))
        x = self.s4(x)
        x = F.relu(self.c5(x))  # num_examples x 120 x 1 x 1
        x = x.reshape(x.shape[0], -1)  # --> num_examples x 120
        x = F.relu(self.f6(x))
        x = self.out(x)
        return x


# x = torch.randn(64, 1, 32, 32)
# model = LeNet()
# print(model(x).shape) # torch.Size([64, 10])


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
in_channels = 1
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5

# Transform with Padding
transform = transforms.Compose([
    transforms.Pad(2),  # Pad the image to 32x32
    transforms.ToTensor()
])

# Load Data
train_dataset = datasets.MNIST(
    root='dataset/', train=True, transform=transform, download=True)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(
    root='dataset/', train=False, transform=transform, download=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)
