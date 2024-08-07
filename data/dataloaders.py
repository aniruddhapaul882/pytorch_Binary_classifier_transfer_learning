import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import os
from PIL import Image

from config import Config as cfg

import os
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

class DataLoaderSetup:
    def __init__(self, data_dir=cfg.DATA_DIR, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(cfg.RESIZE_SIZE),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(cfg.RESIZE_SIZE),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x),
                                                      self.data_transforms[x])
                              for x in ['train', 'val']}
        self.dataloaders = {x: DataLoader(self.image_datasets[x], batch_size=self.batch_size,
                                         shuffle=True, num_workers=self.num_workers)
                            for x in ['train', 'val']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes

    def get_dataloaders(self):
        return self.dataloaders

    def get_dataset_sizes(self):
        return self.dataset_sizes

    def get_class_names(self):
        return self.class_names

    def get_device(self):
        return self.device
