import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
from tqdm import tqdm
import numpy as np
import warnings

warnings.filterwarnings("ignore", message="The default value of the antialias parameter of all the resizing transforms")


# Argument parser for command line options
parser = argparse.ArgumentParser(description="Train a VGG model on chest X-ray images")
parser.add_argument('--csv_file', type=str, default='/home/adam/final_project/APS360-Project/ChestX-Ray/data/image_labels.csv', help='Path to the CSV file containing image paths and labels')
parser.add_argument('--image_folder', type=str, default='/home/adam/final_project/APS360-Project/ChestX-Ray/data/nih-chest-xray-dataset/balanced', help='Path to the folder containing images')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train the model')
parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for the DataLoader')
parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use for training, e.g., "0,1,2"')

args = parser.parse_args()

# Parse the GPU string to a list of integers
gpu_ids = list(map(int, args.gpus.split(',')))

# Paths
csv_file = args.csv_file
image_folder = args.image_folder
device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

# Hyperparameters
batch_size = args.batch_size
learning_rate = args.learning_rate
num_epochs = args.num_epochs
num_workers = args.num_workers
num_classes = len(os.listdir(image_folder))

# Create a mapping from class names to indices
class_names = sorted(os.listdir(image_folder))  # Ensure consistent ordering
class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}

# Custom Dataset Class
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_folder, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_folder = image_folder
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 1], self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = self.data.iloc[idx, 1]
        
        if self.transform:
            image = self.transform(image)
        
        # Convert label to an integer using class_to_idx
        label = class_to_idx[label]
        
        return image, label

# Define rigorous data augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize to slightly larger size for later cropping
    transforms.ToTensor(),  # Convert image to Tensor
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop to 224x224 with scaling
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.RandomRotation(15),  # Random rotation within a range of degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random changes in brightness, contrast, saturation, and hue
    transforms.RandomAffine(translate=(0.1, 0.1), degrees=0),  # Slight translation
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.1), ratio=(0.3, 3.3), value='random'),  # Random erasing
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize to ImageNet means and stds
])

# Define a simpler transform for the test set
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize directly to the target size
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load CSV and create train-test split
data = pd.read_csv(csv_file)
train_data, test_data = train_test_split(data, test_size=0.2, stratify=data['Class Name'], random_state=42)

# Save the train and test splits
train_data.to_csv('vgg_train_data.csv', index=False)
test_data.to_csv('vgg_test_data.csv', index=False)

# Create dataset objects
train_dataset = ChestXrayDataset(csv_file='vgg_train_data.csv', image_folder=image_folder, transform=train_transform)
test_dataset = ChestXrayDataset(csv_file='vgg_test_data.csv', image_folder=image_folder, transform=test_transform)

# Create data loaders with num_workers
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

# Load pre-trained VGG model
model = models.vgg16(pretrained=False)

# Replace the final fully connected layer
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, num_classes)

# Ensure the model is on the correct device
model = model.to(device)

# Enable multi-GPU if available and specified
if len(gpu_ids) > 1 and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids=gpu_ids)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Lists to store training and validation loss and accuracy
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# Early stopping parameters
patience = 3 # Number of epochs to wait after last improvement
min_delta = 0.001  # Minimum change in the monitored quantity to qualify as an improvement
best_val_loss = float('inf')
patience_counter = 0

# Training the model with tqdm progress bar
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Add tqdm to track progress
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
    
    # Evaluate on the test set
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(test_loader)
    epoch_acc = 100 * correct / total
    test_losses.append(epoch_loss)
    test_accuracies.append(epoch_acc)
    
    print(f'Validation Loss: {epoch_loss:.4f}, Validation Accuracy: {epoch_acc:.2f}%')
    
	# Early stopping check
    if epoch_loss < best_val_loss - min_delta:
        best_val_loss = epoch_loss
        patience_counter = 0  # Reset patience counter if validation loss improves
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print(f'Early stopping triggered after epoch {epoch + 1}')
        break

# Save training and validation curves as PDF
plt.figure()
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), test_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('VGG Training and Validation Loss')
plt.savefig('vgg_loss_curve.pdf')

plt.figure()
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), test_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('VGG Training and Validation Accuracy')
plt.savefig('vgg_accuracy_curve.pdf')

# Confusion matrix and ROC curve
model.eval()
all_labels = []
all_preds = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 7))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('VGG Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(num_classes)
plt.xticks(tick_marks, class_names, rotation=45)
plt.yticks(tick_marks, class_names)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('vgg_confusion_matrix.pdf')

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(np.array(all_labels) == i, np.array(all_preds) == i)
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure()
for i in range(num_classes):
    plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('VGG Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('vgg_roc_curve.pdf')

print('Training and evaluation complete. Curves and matrices saved as PDF.')