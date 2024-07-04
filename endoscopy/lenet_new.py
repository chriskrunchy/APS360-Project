import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

# Define labels and directory
labels = ['Diverticulosis', 'Neoplasm', 'Peritonitis', 'Ureters']
DIR = '/home/adam/endoscopy_data'

# Prepare data
IMG_SIZE = 224
X = []
y = []

for i in labels:
    folderPath = os.path.join(DIR, i)
    for j in tqdm(os.listdir(folderPath)):
        img = Image.open(os.path.join(folderPath, j)).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        X.append(np.array(img))
        y.append(i)

X = np.array(X)
y = np.array(y)


# Train-test split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=101)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=101)

print('X_train shape', X_train.shape)
print('y_train shape', y_train.shape)
print('X_val shape', X_val.shape)
print('y_val shape', y_val.shape)
print('X_test shape', X_test.shape)
print('y_test shape', y_test.shape)

# Convert labels to indices
y_train = [labels.index(i) for i in y_train]
y_val = [labels.index(i) for i in y_val]
y_test = [labels.index(i) for i in y_test]

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)
print(X_test.shape)
print(y_test.shape)

class GastroDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(contrast=0.2),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = GastroDataset(X_train, y_train, transform)
val_dataset = GastroDataset(X_val, y_val, transform)
test_dataset = GastroDataset(X_test, y_test, transform)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Set device 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self, num_classes=4):
        super(LeNet, self).__init__()
        self.c1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.s2 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.s4 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5, 5), stride=(1, 1), padding=(0, 0))
        self.s6 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.f6 = nn.Linear(in_features=120 * 24 * 24, out_features=84)  # Adjusted feature size
        self.out = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = torch.relu(self.c1(x))
        x = self.s2(x)
        x = torch.relu(self.c3(x))
        x = self.s4(x)
        x = torch.relu(self.c5(x))
        x = self.s6(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layer
        x = torch.relu(self.f6(x))
        x = self.out(x)
        return x

model = LeNet(num_classes=len(labels)).to(device)
print(model)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


num_epochs = 100
best_accuracy = 0.0

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training loop with tqdm
    train_loader_tqdm = tqdm(train_loader, desc=f'Epoch [{epoch+1}/{num_epochs}]', unit='batch')
    for train_images, train_labels in train_loader_tqdm:
        train_images, train_labels = train_images.to(device), train_labels.to(device)  # Move images and labels to device
        optimizer.zero_grad()
        outputs = model(train_images)
        loss = criterion(outputs, train_labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += train_labels.size(0)
        correct += (predicted == train_labels).sum().item()
        train_accuracy = 100 * correct / total

        train_loader_tqdm.set_postfix(loss=running_loss/total, accuracy=train_accuracy)

    train_losses.append(running_loss / total)
    train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    val_running_loss = 0.0

    with torch.no_grad():
        for val_images, val_labels in val_loader:
            val_images, val_labels = val_images.to(device), val_labels.to(device)  # Move val_images and val_labels to device
            val_outputs = model(val_images)
            val_loss = criterion(val_outputs, val_labels)
            val_running_loss += val_loss.item()
            _, val_predicted = torch.max(val_outputs.data, 1)
            val_total += val_labels.size(0)
            val_correct += (val_predicted == val_labels).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_losses.append(val_running_loss / val_total)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/total:.4f}, Accuracy: {train_accuracy:.4f}')
    print(f'Validation Loss: {val_running_loss/val_total:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    if val_accuracy > best_accuracy:
        best_accuracy = val_accuracy
        torch.save(model.state_dict(), 'lenet_best_model.pth')

print(f"labels: {labels}")

model.load_state_dict(torch.load('lenet_best_model.pth'))
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for test_images, test_labels in tqdm(test_loader, desc='Testing', unit='batch'):
        test_images, test_labels = test_images.to(device), test_labels.to(device)  # Move test_images and test_labels to device
        test_outputs = model(test_images)
        _, test_predicted = torch.max(test_outputs.data, 1)
        all_labels.extend(test_labels.cpu().numpy())
        all_preds.extend(test_predicted.cpu().numpy())

# Ensure the number of unique classes in all_labels and all_preds
print(f"Unique classes in all_labels: {np.unique(all_labels)}")
print(f"Unique classes in all_preds: {np.unique(all_preds)}")

# Check for correct target names and their count
print(f"Target names: {labels}")
print(f"Number of target names: {len(labels)}")

print(classification_report(all_labels, all_preds, target_names=labels))
print(confusion_matrix(all_labels, all_preds))

classification_rep = classification_report(all_labels, all_preds, target_names=labels, output_dict=True)
print(classification_rep)
print(confusion_matrix(all_labels, all_preds))

cm = confusion_matrix(all_labels, all_preds)

# Plotting the confusion matrix and classification report
plt.figure(figsize=(14, 10))

plt.subplot(2, 1, 1)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

plt.subplot(2, 1, 2)
report_df = pd.DataFrame(classification_rep).transpose()
sns.heatmap(report_df.iloc[:-1, :-1].T, annot=True, cmap="Blues")
plt.title('Classification Report')

plt.tight_layout()
plt.savefig('lenet_confusion_matrix_classification_report.png')
plt.show()

# Plotting the loss and accuracy
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('lenet_loss_accuracy_plot.png')
plt.show()

