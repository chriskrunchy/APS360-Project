# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image
import os


# LeNet architecture
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6,
                            kernel_size=5, stride=1, padding=0)
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16,
                            kernel_size=5, stride=1, padding=0)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)  # same as s2
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120,
                            kernel_size=5, stride=1, padding=0)
        self.f6 = nn.Linear(in_features=120*1*1, out_features=84)
        self.out = nn.Linear(in_features=84, out_features=3)  # 3 classes

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


# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Hyperparameters
in_channels = 1
num_classes = 3
learning_rate = 0.001
batch_size = 64
num_epochs = 5


# Transform with Padding
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])


# Load Data
dataset = CTDataset(
    root_dir='data/cancer-tumor-aneurysm-ct-dataset/', transform=transform)

# Split dataset into training and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size, shuffle=False)


# Initiate Network
model = LeNet().to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()  # for classification
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# Train Network
for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        scores = model(data)
        loss = criterion(scores, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent or adam step
        optimizer.step()


# Check accuracy on training & test to see how good our model works
def check_accuracy(loader, model):
    if loader.dataset == train_dataset:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        accuracy = float(num_correct) / float(num_samples) * 100
        print(f'Got {num_correct} / {num_samples} with accuracy {accuracy:.2f}%')

    model.train()


check_accuracy(train_loader, model)
check_accuracy(test_loader, model)
