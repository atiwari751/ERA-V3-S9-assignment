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

# Define transformations
transform = transforms.Compose([
    transforms.Resize(256),  # Resize the smaller side to 256 pixels while keeping aspect ratio
    transforms.CenterCrop(224),  # Then crop to 224x224 pixels from the center
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

# Train dataset and loader
trainset = datasets.ImageFolder(root='/mnt/imagenet/ILSVRC/Data/CLS-LOC/train', transform=transform)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=16, pin_memory=True)

testset = datasets.ImageFolder(root='/mnt/imagenet/ILSVRC/Data/CLS-LOC/val', transform=transform )
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
from tqdm import tqdm

def train(model, device, train_loader, optimizer, criterion, epoch, accumulation_steps=4):
    model.train()
    running_loss = 0.0
    correct = 0
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
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_description(desc=f'Epoch {epoch} | Loss: {running_loss / (batch_idx + 1):.4f} | Accuracy: {100. * correct / total:.2f}%')

        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()

    return 100. * correct / total


# Testing function
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    test_accuracy = 100.*correct/total
    print(f'Test Loss: {test_loss/len(test_loader):.4f}, Accuracy: {test_accuracy:.2f}%')
    return test_accuracy, test_loss/len(test_loader)

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

    for epoch in range(1, 6):  # 20 epochs
        train_accuracy = train(model, device, trainloader, optimizer, criterion, epoch)
        test_accuracy, test_loss = test(model, device, testloader, criterion)
        print(f'Epoch {epoch} | Train Accuracy: {train_accuracy:.2f}% | Test Accuracy: {test_accuracy:.2f}%')  
        
        # Append results for this epoch
        results.append((epoch, train_accuracy, test_accuracy))
        
        if test_loss < best_loss:
            best_loss = test_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_path)
        else: 
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered. Training terminated.")
            break

    # Print the results in a tab-separated format
    print("\nEpoch\tTrain Accuracy\tTest Accuracy")
    for epoch, train_acc, test_acc in results:
        print(f"{epoch}\t{train_acc:.2f}\t{test_acc:.2f}")
