import os
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

from utils.folder2lmdb import ImageFolderLMDB
from config import CONFIG

CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_loaders(train_portion, batch_size, path_to_save_data, logger):
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

    train_data = datasets.CIFAR100(root=path_to_save_data, train=True,
                                  download=True, transform=train_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(train_portion * num_train))

    train_idx, valid_idx = indices[:split], indices[split:]
    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, sampler=train_sampler,
                num_workers=CONFIG["dataloading"]["cifar10"]["num_workers"]
            )

    if train_portion == 1:
        return train_loader, None

    valid_sampler = SubsetRandomSampler(valid_idx)
    val_loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, sampler=valid_sampler,
                num_workers=CONFIG["dataloading"]["cifar10"]["num_workers"]
            )

    return train_loader, val_loader

def get_test_loader(batch_size, path_to_save_data):
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    test_data = datasets.CIFAR100(root=path_to_save_data, train=False,
                                 download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                              shuffle=False, num_workers=CONFIG["dataloading"]["cifar10"]["num_workers"])

    return test_loader

def get_imagenet_loaders(train_portion, batch_size, path_to_save_data, logger):
    traindir = os.path.join(path_to_save_data, "train_lmdb", "train.lmdb")
    #traindir = os.path.join(path_to_save_data, "val_lmdb", "val.lmdb")
    valdir = os.path.join(path_to_save_data, "val_lmdb", "val.lmdb")


    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(
            #    brightness=0.4,
            #    contrast=0.4,
            #    saturation=0.4,
            #    hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    train_data = ImageFolderLMDB(traindir, train_transform, None)
    val_data = ImageFolderLMDB(valdir, val_transform, None)

    train_loader = torch.utils.data.DataLoader(
                train_data, batch_size=batch_size, pin_memory=True, shuffle=True,
                num_workers=CONFIG["dataloading"]["imagenet"]["num_workers"]
            )


    #val_loader = torch.utils.data.DataLoader(
    #            val_data, batch_size=batch_size, pin_memory=True,
    #            num_workers=12
    #        )
    val_loader = None

    return train_loader, val_loader

def get_imagenet_test_loaders(batch_size, path_to_save_data):
    testdir = os.path.join(path_to_save_data, "test_lmdb", "test.lmdb")
    
    test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    test_data = ImageFolderLMDB(testdir, test_transform, None)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                  shuffle=False, pin_memory=True,
                  #num_workers=CONFIG["dataloading"]["num_workers"])
                  num_workers=8)
    #test_loader = None

    return test_loader
