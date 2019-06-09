import os
import time
import copy
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from torch.optim import lr_scheduler


import cv2
import matplotlib.pyplot as plt

from cellcoredataset import CellCoreDataset, RandomCrop, RandomRescale, Rescale

import cellcoredataset
from network import My_Net, init_weights


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                inputs = torch.Tensor(data['image']).unsqueeze(1)
                labels = torch.Tensor(data['gt_map']).unsqueeze(1)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
 
            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # read datasets
    
    datasets_filelist = {'train':[], 'val':[]}
    for ds_type in ['train', 'val']:
        with open(ds_type + ".csv", 'r') as file:
            datasets_filelist[ds_type] = file.read().split('\n')
        del(datasets_filelist[ds_type][-1])

    print('dsf')
    print(len(datasets_filelist['train']))
    print(len(datasets_filelist['val']))


    # train_db = CellCoreDataset(filenamelist=datasets['train'])
    # print(len(train_db))
    # val_db = CellCoreDataset(filenamelist=datasets['val'])
    # print(len(val_db))

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            Rescale( int(1864 / 2) ),
            RandomRescale( 0.9, 1.1 ), 
            RandomCrop(224)
        ]),
        'val': transforms.Compose([
            Rescale( int(1864 / 2) ),
            RandomCrop(224)
        ]),
    }

    image_datasets = {x: CellCoreDataset(filenamelist=datasets_filelist[x],
                        transform=data_transforms[x], output_reduction=4)
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                shuffle=True, num_workers=0)
                for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



    net = My_Net()
    print(net)
    net.apply(init_weights)

    # loss function
    # mean-squared error between the input and the target
    criterion = nn.MSELoss()
    #criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer = optim.Adam(net.parameters(), lr=0.01)
    #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    model = train_model(net, criterion, optimizer, exp_lr_scheduler, num_epochs=15)

    my_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    torch.save(model.state_dict(), 'model_' + my_time + '.pth')