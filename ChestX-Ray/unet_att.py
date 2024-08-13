import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import pandas as pd
from PIL import Image
import os
import timm
from torchsummary import summary
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np


# Define a custom dataset class
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = int(self.img_labels.iloc[idx, 1])

        if self.transform:
            image = self.transform(image)

        return image, label

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = CustomImageDataset(
    annotations_file='/home/adam/final_project/APS360-Project/ChestX-Ray/data/balanced_images.csv', 
    img_dir='/home/adam/final_project/APS360-Project/ChestX-Ray/data/nih-chest-xray-dataset/balanced', 
    transform=transform
)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

# Define the CNN + ViT model
class CNN_ViT(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN_ViT, self).__init__()
        # Pretrained CNN (ResNet50)
        self.cnn = timm.create_model('resnet50', pretrained=True, num_classes=0)
        
        # Vision Transformer (DeiT Small)
        self.vit = timm.create_model('deit_small_patch16_224', pretrained=True, num_classes=0)
        
        # Final Classification Head
        self.fc = nn.Linear(768 + 2048, num_classes)

    def forward(self, x):
        # Pass through CNN
        cnn_features = self.cnn(x)
        
        # Pass through ViT
        vit_features = self.vit(x)
        
        # Concatenate features
        combined_features = torch.cat((cnn_features, vit_features), dim=1)
        
        # Final Classification
        output = self.fc(combined_features)
        return output

# Initialize the model, optimizer, and loss function
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN_ViT(num_classes=10).to(device)
# summary(model, (3, 224, 224))

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop
epochs = 20
best_val_loss = float('inf')

# Initialize lists to store the training and validation metrics
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data)
    
    train_loss = running_loss / len(train_loader.dataset)
    train_acc = correct_predictions.double() / len(train_loader.dataset)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc.item())
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct_predictions += torch.sum(preds == labels.data)
    
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct_predictions.double() / len(val_loader.dataset)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())
    
    print(f"Training Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}")
    print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
    
    # Early stopping and model saving
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        print("Model saved.")

# Load the best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Plot training and validation loss and accuracy
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(12, 6))

# Loss plot
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label='Training Loss')
plt.plot(epochs_range, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy plot
plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label='Training Accuracy')
plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_validation_curves.pdf')

# Confusion matrix
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.pdf')

# ROC curve and AUC
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
n_classes = len(np.unique(all_labels))
labels_binarized = label_binarize(all_labels, classes=[*range(n_classes)])

fpr = dict()
tpr = dict()
roc_auc = dict()

# Compute ROC curve and ROC area for each class
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(labels_binarized[:, i], all_preds == i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve.pdf')

print("Training curves, confusion matrix, and ROC curve saved as PDFs.")