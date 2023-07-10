import os
from tkinter import Tk
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import shutil
import time

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Tk().withdraw()

data_dir = os.path.realpath(os.path.dirname(__file__))

model_path = os.path.join(data_dir, "model.pt")

def create_train_test_folders(root_folder, split_percentage):
    class_names = [name for name in os.listdir(root_folder) if not name.startswith('.') and os.path.isdir(os.path.join(root_folder, name))]

    train_dir = os.path.join(root_folder, 'train')
    os.makedirs(train_dir, exist_ok=True)
    valid_dir = os.path.join(root_folder, 'valid')
    os.makedirs(valid_dir, exist_ok=True)

    for class_name in class_names:
        class_train_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        class_valid_dir = os.path.join(valid_dir, class_name)
        os.makedirs(class_valid_dir, exist_ok=True)

        class_dir = os.path.join(root_folder, class_name)
        all_files = [name for name in os.listdir(class_dir) if
                     not name.startswith('.') and os.path.isfile(os.path.join(class_dir, name))]

        random.shuffle(all_files)
        split_index = int(len(all_files) * split_percentage)
        train_files = all_files[:split_index]
        valid_files = all_files[split_index:]

        for train_file in train_files:
            src_file = os.path.join(class_dir, train_file)
            dst_file = os.path.join(class_train_dir, train_file)
            shutil.copy(src_file, dst_file)
        for valid_file in valid_files:
            src_file = os.path.join(class_dir, valid_file)
            dst_file = os.path.join(class_valid_dir, valid_file)
            shutil.copy(src_file, dst_file)

def train_model(data_dir, device):
    model = models.alexnet(pretrained=True)
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    num_classes = len(train_dataset.classes)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model_path = os.path.join(data_dir, "model.pt")
    valid_dir = os.path.join(data_dir, 'valid')
    if not os.path.exists(valid_dir):
        create_train_test_folders(data_dir, 0.8)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
    print("Data başarılı bir şekilde yaratıldı.")

    num_epochs = int(input("Kaç aşamalı eğitime tabi tutulsun:(Ör: 2)"))
    print("Model eğitiliyor...")
    best_model = None
    best_valid_acc = 0.0
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(train_loader)
        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
            valid_acc = valid_correct / valid_total

            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model = model
        if best_model is not None:
            torch.save(best_model, model_path)

    if best_model is not None:
        torch.save(best_model, model_path)
        print('Best model saved to: {}'.format(model_path))
        epoch_elapsed_time = time.time() - epoch_start_time
    
    elapsed_time = time.time() - start_time
    print(f"Eğitim şu kadar zamanda bitti: ({elapsed_time:.2f} saniye)")

train_model(data_dir, device)