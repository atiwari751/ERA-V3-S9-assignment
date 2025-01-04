import torch
from tqdm import tqdm
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

    return 100. * correct1 / total, 100. * correct5 / total, running_loss / len(train_loader)

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