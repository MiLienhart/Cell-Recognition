import random
import os

from torchvision import transforms, utils

from cellcoredataset import CellCoreDataset

def split_dataset(dataset):
    datasets = {'train':[], 'val':[], 'test':[]}
    for filename in dataset.cell_image_filenames:
        my_rand = random.random()
        if my_rand <= .7:
            datasets['train'].append(filename) 
        elif my_rand <= 1.0:
            datasets['val'].append(filename)         
        else:
            datasets['test'].append(filename) 
    return datasets


if __name__ == "__main__":
    my_root_dir = "/Users/Michelle/Documents/Augsburg_Uni/SS 2019/Bachelorarbeit/Aufnahmen/Aufnahmen_bearbeitet/20190420_Easter_special/Images_Part_1_Adhesion"
    dataset = CellCoreDataset(my_root_dir)
    print(len(dataset))
    datasets = split_dataset(dataset)
    print(datasets)

    train_db = CellCoreDataset(filenamelist=datasets['train'])
    print(len(train_db))
    val_db = CellCoreDataset(filenamelist=datasets['val'])
    print(len(val_db))

    for ds_type in datasets.keys():
        ds = datasets[ds_type]
        with open(ds_type + ".csv", 'w') as file:
            for filename in ds:
                file.write(filename + '\n')




