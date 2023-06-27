import os
import time
import torch
import random
import torchvision
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from datetime import datetime
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.datasets import ImageFolder
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check if GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define constants
resnet_type = "resnet18"
image_size = 224
batch_size = 8
num_epochs = 1
learning_rate = 0.1
num_classes = 4         # Define num_classes 2 if you want to use dataset with 2 classes
start_time = time.time()

# Generate train dataset by number of classes
if num_classes == 2:
    train_dir_data = "./../../data/Train_class_2"
else:
    train_dir_data = "./../../data/Train_class_4"
   
 # Define the transformations for train data before entering the neural network
transform_train = transforms.Compose([
    transforms.Resize((image_size, image_size)),                                    # Resize images to 224x224 pixels
    transforms.RandomCrop(size=image_size),                                         # Randomly crop and resize images to 224x224 pixels
    transforms.RandomHorizontalFlip(),                                              # Randomly flip the images horizontally
    transforms.RandomRotation(30),                                                  # Randomly rotate the images by 30 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
    transforms.ToTensor(),                                                          # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])                 # Normalize the images
])

# Define the transformations for test data before entering the neural network
transform_test = transforms.Compose([
    transforms.Resize((image_size, image_size)),                     # Resize images to 224x224 pixels
    transforms.ToTensor(),                                           # Convert images to tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the images
])

# Create DataLoaders for training, validation, and test datasets
train_dataset = ImageFolder(train_dir_data, transform=transform_train)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Split the dataset into training, validation, and test sets
dataset_size = len(train_dataset)
train_size = int(train_ratio * dataset_size)
val_size = int(val_ratio * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size, test_size])

# Create DataLoaders for managing the data batches
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define ResNet18 model
model = resnet18(pretrained=True)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5)

# Training function
def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        running_loss += loss.item()
        
    accuracy = 100.0 * correct / total
    return running_loss / len(train_loader), accuracy

# Test function for validation and test sets
def test(model, criterion, loader, device):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    accuracy = 100.0 * correct / total
    return running_loss / len(loader), accuracy

# Initialize the best loss variable with infinity
best_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    train_loss, train_accuracy = train(model, criterion, optimizer, train_loader, device)
    test_loss, test_accuracy = test(model, criterion, test_loader, device)
    val_loss, val_accuracy = test(model, criterion, val_loader, device)
    scheduler.step(test_loss) 

    print(f'Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Val Accuracy: {val_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')
    
    # Save model
    if num_classes == 2:
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), './../../models/ResNet_2classes_weights.pth')
    else:
        if test_loss < best_loss:
            best_loss = test_loss
            torch.save(model.state_dict(), './../../models/ResNet_4classes_weights.pth')      

# Set a time and date
now = datetime.now()
timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
print(timestamp)

duration = time.time() - start_time
duration_str = time.strftime("%H:%M:%S", time.gmtime(duration))
print(f"Total duration: {duration_str}")

# Write data to csv file
data = {
    "Timestamp": timestamp,
    "Duration": duration_str,
    "Model type": resnet_type,
    "Dataset use": os.path.basename(train_dir_data),
    "Image Resize": str(image_size)+"*"+str(image_size),
    "Epochs": num_epochs,
    "Learning rate": learning_rate,
    "Batch size": batch_size,
    "Train Accuracy": f"{train_accuracy:.2f}",
    "Validation accuracy": f"{val_accuracy:.2f}",
    "Test Accuracy": f"{test_accuracy:.2f}",
}

# Check if the CSV file already exists
if os.path.isfile("./../../data/csv/model_data.csv"):
    existing_data = pd.read_csv("./../../data/csv/model_data.csv")
    new_data = pd.concat([existing_data, pd.DataFrame(data, index=[0])], ignore_index=True)
else:
    new_data = pd.DataFrame(data, index=[0])

# Save the updated DataFrame to CSV
new_data.to_csv("./../../data/csv/model_data.csv", index=False)