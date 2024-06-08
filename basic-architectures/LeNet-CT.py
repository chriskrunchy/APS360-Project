# Imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms


# Custom dataset for CT images
class CTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = []
        self.labels = []

        class_names = ['aneurysm', 'cancer', 'tumor']
        for label, class_name in enumerate(class_names):
            class_dir = os.path.join(root_dir, 'files', class_name)
            for file_name in os.listdir(class_dir):
                # Assuming images are in .jpg format
                if file_name.endswith('.jpg'):
                    self.image_files.append(os.path.join(class_dir, file_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('L')  # Convert to grayscale
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


# LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.s2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.s4 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))  # same as s2
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120,
                            kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.f6 = nn.Linear(in_features=120, out_features=84)
        # Adjusted for 3 classes
        self.out = nn.Linear(in_features=84, out_features=3)

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


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
in_channels = 1
num_classes = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 5


# Transform with Padding and Resizing
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32
    transforms.ToTensor()
])

# Load Data
train_dataset = CTDataset(
    root_dir='data/cancer-tumor-aneurysm-ct-dataset', transform=transform)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_dataset = CTDataset(
    root_dir='data/cancer-tumor-aneurysm-ct-dataset', transform=transform)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=True)

# Initiate Network
model = LeNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        scores = model(data)
        loss = criterion(scores, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model works
def check_accuracy(loader, model, is_train=True):
    if is_train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        print(
            f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


check_accuracy(train_loader, model, is_train=True)
check_accuracy(test_loader, model, is_train=False)
