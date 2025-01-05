import torch
import torch.nn as nn
import torch.optim as optim
from resnet_model import ResNet50
from data_utils import get_train_transform, get_test_transform, get_data_loaders
from train_test import train, test
from utils import save_checkpoint, load_checkpoint, plot_training_curves, plot_misclassified_samples
from torchsummary import summary

def main():
    # Initialize model, loss function, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet50()
    model = torch.nn.DataParallel(model)
    model = model.to(device)
    summary(model, input_size=(3, 224, 224))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    # Load data
    train_transform = get_train_transform()
    test_transform = get_test_transform()
    trainloader, testloader = get_data_loaders(train_transform, test_transform)

    # Load checkpoint if it exists
    checkpoint_path = "checkpoint.pth"
    try:
        model, optimizer, start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
    except FileNotFoundError:
        print("No checkpoint found, starting from scratch.")
        start_epoch = 1

    # Store results for plotting
    results = []
    learning_rates = []

    # Training loop
    for epoch in range(start_epoch, 26):
        train_accuracy1, train_accuracy5, train_loss = train(model, device, trainloader, optimizer, criterion, epoch)
        test_accuracy1, test_accuracy5, test_loss, misclassified_images, misclassified_labels, misclassified_preds = test(model, device, testloader, criterion)
        print(f'Epoch {epoch} | Train Top-1 Acc: {train_accuracy1:.2f} | Test Top-1 Acc: {test_accuracy1:.2f}')

        # Append results for this epoch
        results.append((epoch, train_accuracy1, train_accuracy5, test_accuracy1, test_accuracy5, train_loss, test_loss))
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, test_loss, checkpoint_path)

    # Extract results for plotting
    epochs = [r[0] for r in results]
    train_acc1 = [r[1] for r in results]
    train_acc5 = [r[2] for r in results]
    test_acc1 = [r[3] for r in results]
    test_acc5 = [r[4] for r in results]
    train_losses = [r[5] for r in results]
    test_losses = [r[6] for r in results]

    # Plot training curves
    plot_training_curves(epochs, train_acc1, test_acc1, train_acc5, test_acc5, train_losses, test_losses, learning_rates)

    # Plot misclassified samples
    '''
    plot_misclassified_samples(misclassified_images, misclassified_labels, misclassified_preds, classes=['class1', 'class2', ...])  # Replace with actual class names
    '''

if __name__ == '__main__':
    main() 
