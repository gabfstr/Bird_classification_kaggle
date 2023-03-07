import zipfile
import os

import torchvision.transforms as transforms

# once the images are loaded, how do we pre-process them before being passed into the network
# by default, we resize the images to 64 x 64 in size
# and normalize them to mean = 0 and standard-deviation = 1 based on statistics collected from
# the training set
data_transforms = {
    "train" : transforms.Compose([
        transforms.Resize(512),
        transforms.RandomRotation(50),
        transforms.RandomResizedCrop(size=512),
        transforms.ColorJitter(brightness=0.1,contrast=0.1,saturation=0.1,hue=0),   
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]),
    "val" : transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    "test" : transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(size=512),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}




