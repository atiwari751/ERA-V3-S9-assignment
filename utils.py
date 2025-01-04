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

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

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