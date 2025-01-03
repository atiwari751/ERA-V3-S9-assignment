import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from resnet_model import ResNet50
from tqdm import tqdm
from torchvision import datasets
from checkpoint import save_checkpoint, load_checkpoint
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

# Define transformations
train_transform = A.Compose([
    A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(3/4, 4/3), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

test_transform = A.Compose([
    A.Resize(height=256, width=256),
    A.CenterCrop(height=224, width=224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Train dataset and loader
trainset = datasets.ImageFolder(root='/mnt/imagenet/ILSVRC/Data/CLS-LOC/train', transform=lambda img: train_transform(image=np.array(img))['image'])
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)

testset = datasets.ImageFolder(root='/mnt/imagenet/ILSVRC/Data/CLS-LOC/val', transform=lambda img: test_transform(image=np.array(img))['image'])
testloader = DataLoader(testset, batch_size=1000, shuffle=False, num_workers=16, pin_memory=True)

# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50()
model = torch.nn.DataParallel(model)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

# Training function
from torch.amp import autocast

def train(model, device, train_loader, optimizer, criterion, epoch, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    correct1 = 0
    correct5 = 0
    total = 0
    pbar = tqdm(train_loader)

    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        with autocast(device_type='cuda'):
            outputs = model(inputs)
            loss = criterion(outputs, targets) / accumulation_steps

        loss.backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        running_loss += loss.item() * accumulation_steps
        _, predicted = outputs.topk(5, 1, True, True)
        total += targets.size(0)
        correct1 += predicted[:, :1].eq(targets.view(-1, 1).expand_as(predicted[:, :1])).sum().item()
        correct5 += predicted.eq(targets.view(-1, 1).expand_as(predicted)).sum().item()

        pbar.set_description(desc=f'Epoch {epoch} | Loss: {running_loss / (batch_idx + 1):.4f} | Top-1 Acc: {100. * correct1 / total:.2f} | Top-5 Acc: {100. * correct5 / total:.2f}')

        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    return 100. * correct1 / total, 100. * correct5 / total, running_loss / len(train_loader)

# Testing function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct1 = 0
    correct5 = 0
    total = 0
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.topk(5, 1, True, True)
            total += targets.size(0)
            correct1 += predicted[:, :1].eq(targets.view(-1, 1).expand_as(predicted[:, :1])).sum().item()
            correct5 += predicted.eq(targets.view(-1, 1).expand_as(predicted)).sum().item()

            # Collect misclassified samples
            for i in range(inputs.size(0)):
                if targets[i] not in predicted[i, :1]:
                    misclassified_images.append(inputs[i].cpu())
                    misclassified_labels.append(targets[i].cpu())
                    misclassified_preds.append(predicted[i, :1].cpu())

    test_accuracy1 = 100. * correct1 / total
    test_accuracy5 = 100. * correct5 / total
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Top-1 Accuracy: {test_accuracy1:.2f}, Top-5 Accuracy: {test_accuracy5:.2f}')
    return test_accuracy1, test_accuracy5, test_loss / len(test_loader), misclassified_images, misclassified_labels, misclassified_preds

# Main execution
if __name__ == '__main__':
    # Early stopping parameters and checkpoint path
    checkpoint_path = "checkpoint.pth"
    best_loss = float('inf')
    patience = 5
    patience_counter = 0
    # Load checkpoint if it exists to resume training
    try:
        model, optimizer, best_test_accuracy = load_checkpoint(model, optimizer, checkpoint_path)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")

    # Store results for each epoch
    results = []
    learning_rates = []

    for epoch in range(1, 6):  # 20 epochs
        train_accuracy1, train_accuracy5, train_loss = train(model, device, trainloader, optimizer, criterion, epoch)
        test_accuracy1, test_accuracy5, test_loss, misclassified_images, misclassified_labels, misclassified_preds = test(model, device, testloader, criterion)
        print(f'Epoch {epoch} | Train Top-1 Acc: {train_accuracy1:.2f} | Train Top-5 Acc: {train_accuracy5:.2f} | Test Top-1 Acc: {test_accuracy1:.2f} | Test Top-5 Acc: {test_accuracy5:.2f}')  
        
        # Append results for this epoch
        results.append((epoch, train_accuracy1, train_accuracy5, test_accuracy1, test_accuracy5, train_loss, test_loss))
        learning_rates.append(optimizer.param_groups[0]['lr'])
        
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_path)
        else: 
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered. Training terminated.")
            break

    # Print the Top-1 accuracy results in a tab-separated format
    print("\nEpoch\tTrain Top-1 Accuracy\tTest Top-1 Accuracy")
    for epoch, train_acc1, test_acc1, *_ in results:
        print(f"{epoch}\t{train_acc1:.2f}\t{test_acc1:.2f}")

    # Plotting
    epochs = [r[0] for r in results]
    train_acc1 = [r[1] for r in results]
    train_acc5 = [r[2] for r in results]
    test_acc1 = [r[3] for r in results]
    test_acc5 = [r[4] for r in results]
    train_losses = [r[5] for r in results]
    test_losses = [r[6] for r in results]

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_acc1, label='Train Top-1 Acc')
    plt.plot(epochs, test_acc1, label='Test Top-1 Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Top-1 Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_acc5, label='Train Top-5 Acc')
    plt.plot(epochs, test_acc5, label='Test Top-5 Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Top-5 Accuracy')

    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(2, 2, 4)
    plt.plot(epochs, learning_rates, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.title('Learning Rate')

    plt.tight_layout()
    plt.show()

    # Display some misclassified samples
    if misclassified_images:
        print("\nDisplaying some misclassified samples from the last epoch:")
        misclassified_grid = make_grid(misclassified_images[:16], nrow=4, normalize=True, scale_each=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(misclassified_grid.permute(1, 2, 0))
        plt.title("Misclassified Samples")
        plt.axis('off')
        plt.show()
