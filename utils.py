import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

def save_checkpoint(model, optimizer, epoch, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, start_epoch, loss

def plot_training_curves(epochs, train_acc1, test_acc1, train_acc5, test_acc5, train_losses, test_losses, learning_rates):
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

def plot_misclassified_samples(misclassified_images, misclassified_labels, misclassified_preds, classes):
    if misclassified_images:
        print("\nDisplaying some misclassified samples:")
        misclassified_grid = make_grid(misclassified_images[:16], nrow=4, normalize=True, scale_each=True)
        plt.figure(figsize=(8, 8))
        plt.imshow(misclassified_grid.permute(1, 2, 0))
        plt.title("Misclassified Samples")
        plt.axis('off')
        plt.show() 

def find_lr(model, criterion, optimizer, train_loader, num_epochs=1, start_lr=1e-7, end_lr=10, lr_multiplier=1.1):
    """
    Find the optimal learning rate using LR Finder.
    
    Args:
    - model: The model to train
    - criterion: Loss function (e.g., CrossEntropyLoss)
    - optimizer: Optimizer (e.g., SGD)
    - train_loader: DataLoader for training data
    - num_epochs: Number of epochs to run the LR Finder (typically 1-2)
    - start_lr: Starting learning rate for the experiment
    - end_lr: Maximum learning rate (used for scaling)
    - lr_multiplier: Factor by which the learning rate is increased every batch
    
    Returns:
    - A plot of loss vs learning rate
    """
    lrs = []
    losses = []
    avg_loss = 0.0
    batch_count = 0
    
    lr = start_lr
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.param_groups[0]['lr'] = lr  # Set the learning rate
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            avg_loss += loss.item()
            batch_count += 1
            lrs.append(lr)
            losses.append(loss.item())
            
            # Increase the learning rate for next batch
            lr *= lr_multiplier
        
        avg_loss /= batch_count
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
    
    # Plot the loss vs learning rate
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.show()

