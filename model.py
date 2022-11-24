import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20 
# model = models.resnet50(pretrained=True)
# #adpating the size of the fully connected layer
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, nclasses)

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, nclasses)

#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2(x), 2))
#         x = F.relu(F.max_pool2d(self.conv3(x), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

class Net(nn.Module):

    def __init__(self,):
        super(Net,self).__init__()

        self.resnet = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        self.num_ftrs = self.resnet.fc.out_features
        self.fc = nn.Linear(self.num_ftrs, nclasses)

        # freeze the resnet layers
        #for param in self.resnet.parameters():
        #    param.requires_grad = True
            

    def forward(self,x):
        x = self.resnet(x)
        return self.fc(x)