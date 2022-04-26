# %% [markdown]
# - Code taken from https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
# 

# %%
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import cv2
import pandas as pd

cudnn.benchmark = True
plt.ion() 

# %%
data = pd.read_csv("../Dataset/matchedpairs_finalds.csv")
data = data.sample(frac=1).reset_index(drop = True)
total_labels = data.OriginalEntity.unique()

data.loc[:, "Labels"] = data.OriginalEntity.apply(lambda x: np.where(x == total_labels)[0][0])

# %%
class ImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data,root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.entityds = data.copy()
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.entityds)

    def __getitem__(self, idx):
        print(idx)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.entityds.loc[idx, "Image"])
        image = cv2.imread(img_name)
        if self.transform:
            image = self.transform(image)

        labels = self.entityds.loc[idx, "Labels"]
        #sample = {'image': image, 'entity': labels}

        return image, labels

# %%
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.2)
datadict = {}
datadict["train"] = train.reset_index(drop = True)
datadict["val"] = test.reset_index(drop = True)

# %%
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.4372, 0.4457, 0.4807],[0.3319, 0.3318, 0.3384])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.4372, 0.4457, 0.4807],[0.3319, 0.3318, 0.3384])
    ]),
}

data_dir = '../Dataset/matchedpairs_finalds.csv'
image_datasets = {x: ImageDataset(data=datadict[x],
                                    root_dir='../Dataset/images/'
                                    ,transform= data_transforms[x])
                                    for x in ['train', 'val']}
                                    
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=1)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = data.OriginalEntity.unique()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
next(iter(dataloaders["train"]))

# %%
for i_batch, sample_batched in dataloaders["train"]:
    print(i_batch, sample_batched)

# %%
# Get the mean band variance of entire dataset
print(image_datasets["train"][0]["image"].shape)
'''
total_samples = len(image_datasets["train"])
mean = 0
var = 0
for i in range(total_samples):
    sample = image_datasets["train"][i]
    mean += torch.mean(sample["image"], dim = [1,2])
mean = mean / total_samples
print(mean)

for i in range(total_samples):
    image = image_datasets["train"][i]["image"]
    image = image.view(image.size(0),-1)
    var += ((image - mean.unsqueeze(1))**2).sum([1])
var = torch.sqrt(var / (total_samples*224*224))
print(var)
'''

# %% [markdown]
# def imshow(inp, title=None):
#     """Imshow for Tensor."""
#     inp = inp.numpy().transpose((1, 2, 0))
#     mean = np.array([0.485, 0.456, 0.406])
#     std = np.array([0.229, 0.224, 0.225])
#     inp = std * inp + mean
#     inp = np.clip(inp, 0, 1)
#     plt.imshow(inp)
#     if title is not None:
#         plt.title(title)
#     plt.pause(0.001)  # pause a bit so that plots are updated
# 
# 
# # Get a batch of training data
# inputs, classes = next(iter(dataloaders['train']))
# 
# # Make a grid from batch
# out = torchvision.utils.make_grid(inputs)
# 
# imshow(out, title=[class_names[x] for x in classes])

# %%
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

# %%
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

# %%
model_ft = models.resnet18(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False

num_ftrs = model_ft.fc.in_features
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
model_ft.fc = nn.Linear(num_ftrs, len(total_labels))

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# %%
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=25)


