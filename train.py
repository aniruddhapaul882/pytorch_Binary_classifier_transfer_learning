import os
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
from torchvision import models
from data.dataloaders import DataLoaderSetup  # Ensure this imports correctly
from config import Config as cfg
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

def calculate_sample_weights(dataloader):
    class_counts = np.zeros(len(dataloader.dataset.classes))
    
    # Iterate over batches in the dataloader
    for inputs, labels in dataloader:
        # Accumulate counts for each label in the batch
        for label in labels:
            class_counts[int(label.item())] += 1

    # Calculate class weights
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float)
    
    # Calculate sample weights for each item in the dataloader
    sample_weights = [class_weights[int(label.item())] for _, labels in dataloader for label in labels]
    
    return sample_weights, class_counts

def train_model(model, dataloaders, dataset_sizes, device, num_epochs=cfg.NUM_EPOCHS):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=cfg.STEP_SIZE, gamma=cfg.GAMMA)
    
    best_model_wts = model.state_dict()
    best_acc = 0.0
    best_val_loss = float('inf')
    

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.where(outputs > 0.5, 1.0, 0.0)
                    loss = criterion(outputs, labels.float().view(-1, 1))
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.unsqueeze(1))
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = model.state_dict()
        
            if phase == 'val':
                # Save model if validation loss is the lowest seen so far
                if epoch_loss < best_val_loss:
                    best_val_loss = epoch_loss
                    best_model_wts = model.state_dict()
                    torch.save(model.state_dict(), os.path.join(cfg.WEIGHT_DIR, f'{datetime.now()}_full_best.pt'))
                    print(f'Validation loss improved to {epoch_loss:.4f}. Model saved to {cfg.WEIGHT_DIR}')

        exp_lr_scheduler.step()
                
    # print(f'Best val Acc: {best_acc:.4f}')
    print(f'Best val Loss: {best_val_loss:.4f}')
    model.load_state_dict(best_model_wts)
    return model

def main():
    # Load data
    os.makedirs(cfg.WEIGHT_DIR, exist_ok=True)
    os.makedirs(cfg.RESULT_DIR, exist_ok=True)
    data_loader_setup = DataLoaderSetup()
    dataloaders = data_loader_setup.get_dataloaders()
    dataset_sizes = data_loader_setup.get_dataset_sizes()
    device = data_loader_setup.get_device()
    
    # Handle class imbalance using weighted sampling
    sample_weights, class_counts = calculate_sample_weights(dataloaders['train'])
    weighted_sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    dataloaders['train'] = torch.utils.data.DataLoader(dataset=dataloaders['train'].dataset, batch_size=cfg.BATCH_SIZE, sampler=weighted_sampler)
    
    # Define the model (ResNet18 for binary classification)
    model_ft = models.resnet18(weights='IMAGENET1K_V1')
#####Uncomment this part to only finetune the final layer#####
    # for params in model_ft.parameters():
    #     params.requires_grad = False
##############################################################
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    model_ft = model_ft.to(device)
    
    # Train the model
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, device, num_epochs=cfg.NUM_EPOCHS)
    
    # Save the model
    torch.save(model_ft.state_dict(), os.path.join(cfg.WEIGHT_DIR, f'{datetime.now()}_full_last.pt'))

    # Plot Confusion Matrix
    model_ft.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs, labels = inputs.to(device), labels.to(device)
            # print(labels)
            outputs = model_ft(inputs)
            preds = torch.where(outputs > cfg.PRED_THRESH, 1.0, 0.0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
    
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(len(y_pred),"----",len(y_true))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(cfg.RESULT_DIR, 'confusion_matrix.png'))
    plt.show()

if __name__ == '__main__':
    main()
