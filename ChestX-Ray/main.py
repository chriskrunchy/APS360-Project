# import os
# import argparse
# import copy
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split
# from torchvision import transforms
# from sklearn.metrics import accuracy_score, f1_score
# from model2 import MultiConv_Transformer
# from tqdm import tqdm
# from PIL import Image

# class CustomDataset(Dataset):
#     def __init__(self, image_paths, labels, transform=None):
#         self.image_paths = image_paths
#         self.labels = labels
#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx]
#         image = Image.open(img_path).convert('RGB')
#         label = self.labels[idx]
#         if self.transform:
#             image = self.transform(image)
#         return image, label

# def load_data(data_dir, class_names, transforms, batch_size=32, val_split=0.2):
#     image_paths = []
#     labels = []
#     for label, class_name in enumerate(class_names):
#         class_dir = os.path.join(data_dir, class_name)
#         for img_name in os.listdir(class_dir):
#             image_paths.append(os.path.join(class_dir, img_name))
#             labels.append(label)

#     train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=val_split, stratify=labels)

#     train_dataset = CustomDataset(train_paths, train_labels, transforms['train'])
#     val_dataset = CustomDataset(val_paths, val_labels, transforms['val'])

#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

#     return train_loader, val_loader

# def train_model(model, criterion, optimizer, dataloaders, device, num_epochs, patience=5, checkpoint_path='checkpoint.pth'):
#     model.to(device)
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#     epochs_no_improve = 0

#     for epoch in range(num_epochs):
#         print(f'Epoch {epoch}/{num_epochs - 1}')
#         print('-' * 10)

#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 model.train()
#             else:
#                 model.eval()

#             running_loss = 0.0
#             running_corrects = 0

#             for inputs, labels in tqdm(dataloaders[phase], desc=phase, leave=False):
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)

#                 optimizer.zero_grad()

#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _, preds = torch.max(outputs, 1)
#                     loss = criterion(outputs, labels)

#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)

#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

#             print(f'{phase} Loss: {epoch_loss:.4f} Acc: {100 * epoch_acc:.4f}%')

#             if phase == 'val':
#                 if epoch_acc > best_acc:
#                     best_acc = epoch_acc
#                     best_model_wts = copy.deepcopy(model.state_dict())
#                     epochs_no_improve = 0
#                     torch.save(model.state_dict(), checkpoint_path)
#                 else:
#                     epochs_no_improve += 1

#         if epochs_no_improve >= patience:
#             print('Early stopping due to no improvement in validation accuracy')
#             break

#     print('Training complete')
#     print(f'Best val Acc: {100 * best_acc:.4f}%')
#     model.load_state_dict(best_model_wts)
#     return model

# def evaluate_model(model, dataloader, device):
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     accuracy = accuracy_score(all_labels, all_preds)
#     f1 = f1_score(all_labels, all_preds, average='weighted')
#     print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

# def main(args):
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
#     data_dir = args.data_dir
#     class_names = [
#         "Atelectasis",
#         "Cardiomegaly",
#         "Consolidation",
#         "Edema",
#         "Effusion",
#         "Emphysema",
#         "Enlarged Cardiomediastinum",
#         "Fibrosis",
#         "Fracture",
#         "Infiltration",
#         "Lung Lesion",
#         "Lung Opacity",
#         "Mass",
#         "No Finding",
#         "Nodule",
#         "Pleural Other",
#         "Pleural_Thickening",
#         "Pneumonia",
#         "Pneumothorax"
#     ]
#     num_classes = len(class_names)
#     batch_size = args.batch_size
#     num_epochs = args.epochs
#     learning_rate = args.lr

#     data_transforms = {
#         'train': transforms.Compose([
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#         'val': transforms.Compose([
#             transforms.Resize(256),
#             transforms.CenterCrop(224),
#             transforms.ToTensor(),
#             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
#         ]),
#     }

#     train_loader, val_loader = load_data(data_dir, class_names, data_transforms, batch_size)
#     dataloaders = {'train': train_loader, 'val': val_loader}

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = MultiConv_Transformer(num_classes=num_classes)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#     model = train_model(model, criterion, optimizer, dataloaders, device, num_epochs, patience=args.patience, checkpoint_path=args.checkpoint_path)
#     evaluate_model(model, dataloaders['val'], device)

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Train MultiConv Transformer Model')
#     parser.add_argument('--data_dir', type=str, default="/home/adam/final_project/APS360-Project/MultiConv_Transformer/data/images", help='Path to the dataset directory')
#     parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and testing')
#     parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
#     parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
#     parser.add_argument('--gpus', type=str, default="2,3", help='Comma-separated list of GPU IDs to use')
#     parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
#     parser.add_argument('--checkpoint_path', type=str, default='checkpoint.pth', help='Path to save the model checkpoint')

#     args = parser.parse_args()
#     main(args)

import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from tqdm import tqdm
from model2 import MultiConv_Transformer

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'main_checkpoint.pth')
        self.val_loss_min = val_loss

class CustomDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_dir, class_names, transforms, batch_size=32, val_split=0.2, num_workers=4):
    image_paths = []
    labels = []
    for label, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_dir):
            image_paths.append(os.path.join(class_dir, img_name))
            labels.append(label)

    train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=val_split, stratify=labels)

    train_dataset = CustomDataset(train_paths, train_labels, transforms['train'])
    val_dataset = CustomDataset(val_paths, val_labels, transforms['val'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader


def train_model(model, criterion, optimizer, dataloaders, device, num_epochs, patience, scheduler):
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=f'{phase} Epoch {epoch}', leave=False):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_acc_history.append(epoch_acc.item())
                scheduler.step()
            else:
                val_loss_history.append(epoch_loss)
                val_acc_history.append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {100 * epoch_acc:.4f}%')

            if phase == 'val':
                early_stopping(epoch_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    model.load_state_dict(torch.load('main_checkpoint.pth'))
                    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Training complete')
    print(f'Best val Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, train_loss_history, val_loss_history, train_acc_history, val_acc_history

def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Testing", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f'Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}')

def plot_training_curve(train_loss, val_loss, train_acc, val_acc):
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'b', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curve.pdf')
    plt.close()

def plot_confusion_matrix(model, dataloader, class_names, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Confusion Matrix", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    plt.savefig('confusion_matrix.pdf')
    plt.close()
def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    data_dir = args.data_dir
    class_names = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Enlarged Cardiomediastinum",
        "Fibrosis",
        "Fracture",
        "Infiltration",
        "Lung Lesion",
        "Lung Opacity",
        "Mass",
        "No Finding",
        "Nodule",
        "Pleural Other",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax"
    ]
    class_names_X = [
        "Atelectasis",
        "Cardiomegaly",
        "Consolidation",
        "Edema",
        "Effusion",
        "Emphysema",
        "Fibrosis",
        "Hernia",
        "Infiltration",
        "Mass",
        "No Finding",
        "Nodule",
        "Pleural_Thickening",
        "Pneumonia",
        "Pneumothorax"
    ] # 14 + No Finding

    cleaned_images = [
        'Pneumothorax',
        'Lung Opacity',
        'Infiltration',
        'Atelectasis',
        'Effusion',
        'Consolidation',
        'Edema',
        'Pneumonia',
        'No Finding',
        'Cardiomegaly',
    ]# 13282each -> Total = 146102

    num_classes = len(cleaned_images)
    batch_size = args.batch_size
    num_epochs = args.epochs
    learning_rate = args.lr

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    train_loader, val_loader = load_data(data_dir, cleaned_images, data_transforms, batch_size, num_workers=4)
    dataloaders = {'train': train_loader, 'val': val_loader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MultiConv_Transformer(num_classes=num_classes)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[25, 50, 75], gamma=0.1)

    model, train_loss, val_loss, train_acc, val_acc = train_model(
        model, criterion, optimizer, dataloaders, device, num_epochs, patience=5, scheduler=scheduler)
    
    evaluate_model(model, dataloaders['val'], device)
    plot_training_curve(train_loss, val_loss, train_acc, val_acc)
    plot_confusion_matrix(model, dataloaders['val'], cleaned_images, device)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train and Test Baseline Model')
    parser.add_argument('--data_dir', type=str, default="/home/adam/final_project/APS360-Project/MultiConv_Transformer/data/cleaned_images", help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and testing')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training')
    # parser.add_argument('--num_classes', type=int, default=14, help='Number of classes for training')
    parser.add_argument('--lr', type=float, default=0.1, help='Initial learning rate for training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--gpus', type=str, default="0,1", help='Comma-separated list of GPU IDs to use')
    
    args = parser.parse_args()
    main(args)
