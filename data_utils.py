from torchvision import datasets
from torch.utils.data import DataLoader
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_train_transform():
    return A.Compose([
        A.RandomResizedCrop(height=224, width=224, scale=(0.08, 1.0), ratio=(3/4, 4/3), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_test_transform():
    return A.Compose([
        A.Resize(height=256, width=256),
        A.CenterCrop(height=224, width=224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_data_loaders(train_transform, test_transform, batch_size_train=128, batch_size_test=500):
    trainset = datasets.ImageFolder(root='/mnt/imagenet/ILSVRC/Data/CLS-LOC/train', transform=lambda img: train_transform(image=np.array(img))['image'])
    trainloader = DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=8, pin_memory=True)

    testset = datasets.ImageFolder(root='/mnt/imagenet/ILSVRC/Data/CLS-LOC/val', transform=lambda img: test_transform(image=np.array(img))['image'])
    testloader = DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=8, pin_memory=True)

    return trainloader, testloader 
