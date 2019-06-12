import os
import numpy as np
from random import shuffle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
import torch.optim as optim
from skimage import io, transform, draw, color


import cv2
import matplotlib.pyplot as plt

from cellcoredataset import CellCoreDataset, RandomCrop, RandomRescale, Rescale
from network import My_Net
from annotation import get_files

# show training sample with recognized cell cores drawn in
def show_training_sample(image, map):

    """Show image with recognized cell cores drawn in"""
    #img = transform.resize(image, map.shape)
    img = image

    # convert image from gray scale to rgb
    img = color.gray2rgb(img)

    # add ground truth map to image
    img[:,:,0] += transform.resize(map[:,:], image.shape)

    return img


if __name__ == "__main__":
    cuda_no = ':2'
    device = torch.device("cuda"+cuda_no if torch.cuda.is_available() else "cpu")

    code_path = os.path.dirname(os.path.realpath(__file__))
    #PATH = os.path.join(os.path.dirname(code_path),"models/model_2019-06-10_21-17-16.pth")
    PATH = os.path.join(os.path.dirname(code_path),"models/model_2019-06-11_13-38-40.pth")
    
    net = My_Net()
    net.load_state_dict(torch.load(PATH, map_location=device))
    net.to(device)
    print(net)
    net.eval() 


    datasets_filelist = {'train':[], 'val':[]}
    for ds_type in ['train', 'val']:
        with open(ds_type + ".csv", 'r') as file:
            datasets_filelist[ds_type] = file.read().split('\n')
        del(datasets_filelist[ds_type][-1])

    print(len(datasets_filelist['train']))
    print(len(datasets_filelist['val']))

    data_transforms = {
        'train': transforms.Compose([
            Rescale( int(1864 / 2) ),
            RandomRescale( 0.9, 1.1 ), 
            RandomCrop(446)
        ]),
        'val': transforms.Compose([
            Rescale( int(1864 / 2) )#,
            #RandomCrop(224)
        ]),
    }

    image_datasets = {x: CellCoreDataset(filenamelist=datasets_filelist[x],
                        transform=data_transforms[x], output_reduction=4)
                    for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

    # load data set without annotations
    my_root_dir = "/Users/Michelle/Documents/Augsburg_Uni/SS 2019/Bachelorarbeit/Aufnahmen_src/Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_2_DcAMP"

    my_file_list = sorted(get_files(my_root_dir))
    shuffle(my_file_list)
    test_image_dataset = CellCoreDataset(filenamelist=my_file_list,
                        transform=data_transforms['val'], output_reduction=4)
    
    for data in test_image_dataset: #image_datasets['val']:
        inputs = torch.Tensor(data['image']).unsqueeze(0).unsqueeze(0)
        labels = torch.Tensor(data['gt_map']).unsqueeze(0).unsqueeze(0)

        inputs = inputs.to(device)
        labels = labels.to(device)
        print(inputs.shape)
        outputs = net(inputs)
        print('outputs:',outputs.shape)
        print('labels', labels.shape)

        input_image = inputs.squeeze().detach().numpy()
        my_output = outputs.squeeze().detach().numpy()
        my_min = np.min(my_output)
        my_max = np.max(my_output) 
        gt_map = labels.squeeze().numpy()
        plt.imshow(inputs.squeeze().detach().numpy(), cmap='gray')
        
        bild1 = show_training_sample(input_image, my_output)
        bild2 = show_training_sample(input_image, gt_map)
        ax = plt.subplot(1, 2, 1)
        plt.imshow(bild1)
        ax = plt.subplot(1, 2, 2)
        plt.imshow(bild2)

        print(my_min, my_max)

        plt.show()  # pause a bit so that plots are updated


