import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

# Load pretrained ResNet-50
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)  # Adjust for your dataset
model = model.to('cuda')

# Define optimizer and loss function
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# Prepare dataset and DataLoader
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
train_dataset = datasets.ImageFolder(root='/path/to/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

# Set One-Cycle LR scheduler
epochs = 10
steps_per_epoch = len(train_loader)
lr_max = 1e-3  # Adjust based on LR Finder or task size

scheduler = OneCycleLR(optimizer, max_lr=lr_max, epochs=epochs, steps_per_epoch=steps_per_epoch)

# Training loop
for epoch in range(epochs):
    model.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to('cuda'), labels.to('cuda')

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate using One-Cycle policy

    print(f"Epoch {epoch+1}/{epochs} completed.")

