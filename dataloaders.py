import torch
import torch.utils.data
from torchvision import transforms
import os
import glob
import cv2
from torch.utils.data import DataLoader

# Assuming you have `default_config` in your `config` directory
from config.default_config import Config as cfg


class harnessData(torch.utils.data.Dataset):
    
    def __init__(self, phase, num_workers=cfg.NUM_WORKERS):
        
        self.data_dir = cfg.DATA_DIR  # Directory where data is stored
        self.phase = phase
        # self.batch_size = batch_size
        self.num_workers = num_workers
        self.phase = phase
        # Define transformations based on phase
        self.transform = self.get_transform()

        # Initialize list to store image paths
        self.img_list = []
        
        # Extract class names from directory names
        self.class_names = sorted([d for d in os.listdir(os.path.join(self.data_dir, self.phase)) if os.path.isdir(os.path.join(self.data_dir, self.phase, d))])
        self.idx2_label = {ids: label for ids, label in enumerate(self.class_names)}
        self.label2_idx = {value: key for key, value in self.idx2_label.items()}

        # Populate image list with file paths
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, self.phase, class_name)
            for img_path in glob.glob(os.path.join(class_dir, '*.jpg')):
                self.img_list.append(img_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_list[idx])
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
        if self.transform:
            transformed_image = self.transform(rgb_image)
        else:
            transformed_image = self.val_transform(rgb_image)  # Handle validation transformation if needed

        # Extract label from image path and convert to tensor
        label_name = os.path.basename(os.path.dirname(self.img_list[idx]))
        label = torch.tensor(self.label2_idx[label_name], dtype=torch.long)  # Use dtype=torch.long for classification labels

        return transformed_image, label

    def get_transform(self):
        
        if self.phase == 'train':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(cfg.RESIZE_SIZE),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        elif self.phase == 'val':
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(cfg.RESIZE_SIZE),
                # transforms.CenterCrop(224),
                # transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            raise ValueError("Phase must be 'train' or 'val'")

    # @classmethod
    # def get_dataloaders(cls, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS):
        
    #     dataloaders = {
    #         'train': torch.utils.data.DataLoader(
    #             dataset=cls(phase='train', batch_size=batch_size, num_workers=num_workers),
    #             batch_size=batch_size,
    #             shuffle=True,
    #             num_workers=num_workers
    #         ),
    #         'val': torch.utils.data.DataLoader(
    #             dataset=cls(phase='val', batch_size=batch_size, num_workers=num_workers),
    #             batch_size=batch_size,
    #             shuffle=False,
    #             num_workers=num_workers
    #         )
    #     }

    #     dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}
    #     class_names = dataloaders['train'].dataset.class_names
    #     print(class_names)
    #     return dataloaders, dataset_sizes, class_names

# # Example usage
# if __name__ == "__main__":
#     dataloaders, dataset_sizes, class_names = harnessData.get_dataloaders()

#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # You can now use dataloaders, dataset_sizes, and class_names in your training loop


train_loader = DataLoader(dataset = harnessData('train'),batch_size=8,shuffle=True)
print(next(iter(train_loader))[0].shape)
