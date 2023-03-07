import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

nclasses = 20 
# model = models.resnet50(pretrained=True)
# #adpating the size of the fully connected layer
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs, nclasses)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 20, kernel_size=5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, nclasses)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    

class ResNet50(nn.Module):
    """
    ResNet-50 model with frozen first 6 layers
    """
    def __init__(self,):
        super(ResNet50,self).__init__()

        self.resnet = models.resnet50(weights="ResNet50_Weights.IMAGENET1K_V2")
        self.in_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                nn.Linear(self.in_ftrs, nclasses)
                )

        ct = 0
        for child in self.resnet.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        # This freezes layers 1-6 in the total 10 layers of Resnet50

    def forward(self,x):
        return self.resnet(x)
    

    
class Attention(torch.nn.Module):
    """
    Attention block for CNN model.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(Attention, self).__init__()
        self.conv_depth = torch.nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=in_channels)
        self.conv_point = torch.nn.Conv2d(out_channels, out_channels, kernel_size=(1, 1))
        self.bn = torch.nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)
        self.activation = torch.nn.Tanh()
    def forward(self, inputs):
        x, output_size = inputs
        x = F.adaptive_max_pool2d(x, output_size=output_size)
        x = self.conv_depth(x)
        x = self.conv_point(x)
        x = self.bn(x)
        x = self.activation(x) + 1.0
        return x


class ResNet50Attention(torch.nn.Module):
    """
    Attention-enhanced ResNet-50 model
    """
    weights_loader = staticmethod(models.resnet50)
    def __init__(self, num_classes=nclasses, pretrained=True, use_attention=True):
        super(ResNet50Attention, self).__init__()
        net = self.weights_loader(weights="ResNet50_Weights.DEFAULT")
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.use_attention = use_attention
        net.fc = torch.nn.Linear(
            in_features=net.fc.in_features,
            out_features=num_classes,
            bias=net.fc.bias is not None
        )
        self.net = net
        if self.use_attention:
            self.att1 = Attention(in_channels=64, out_channels=64,     kernel_size=(3, 5), padding=(1, 2))
            self.att2 = Attention(in_channels=64, out_channels=128, kernel_size=(5, 3), padding=(2, 1))
            self.att3 = Attention(in_channels=128, out_channels=256, kernel_size=(3, 5), padding=(1, 2))
            self.att4 = Attention(in_channels=256, out_channels=512, kernel_size=(5, 3), padding=(2, 1))
            if pretrained:
                self.att1.bn.weight.data.zero_()
                self.att1.bn.bias.data.zero_()
                self.att2.bn.weight.data.zero_()
                self.att2.bn.bias.data.zero_()
                self.att3.bn.weight.data.zero_()
                self.att3.bn.bias.data.zero_()
                self.att4.bn.weight.data.zero_()
                self.att4.bn.bias.data.zero_()
    def _forward(self, x):
        return self.net(x)
    
    def _forward_att(self, x):
        print("1 : ",x.shape)
        x = self.net.conv1(x)
        print("2 : ",x.shape)
        x = self.net.bn1(x)
        print("3 : ",x.shape)
        x = self.net.relu(x)
        print("4 : ",x.shape)
        x = self.net.maxpool(x)
        print("5 : ",x.shape)
        x_a = x.clone()
        print(self.net.layer1)
        x = self.net.layer1(x)
        print("6 : ",x.shape)
        print("x.shape : ",x.shape)
        print("\nself.att1((x_a, x.shape[-2:])) : \n",self.att1((x_a, x.shape[-2:])).shape)
        x = x * self.att1((x_a, x.shape[-2:]))
        x_a = x.clone()
        x = self.net.layer2(x)
        x = x * self.att2((x_a, x.shape[-2:]))
        x_a = x.clone()
        x = self.net.layer3(x)
        x = x * self.att3((x_a, x.shape[-2:]))
        x_a = x.clone()
        x = self.net.layer4(x)
        x = x * self.att4((x_a, x.shape[-2:]))
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)
        return x
    
    def forward(self, x):
        return self._forward_att(x) if self.use_attention else    self._forward(x)
    



class ResNet101(nn.Module):

    def __init__(self,):
        super(ResNet101,self).__init__()

        self.resnet = models.resnet101(weights="ResNet101_Weights.IMAGENET1K_V2")
        self.in_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                nn.Linear(self.in_ftrs, nclasses)
                )

        ct = 0
        for child in self.resnet.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        # This freezes layers 1-6 in the total 10 layers of Resnet101
                
    def forward(self,x):
        return self.resnet(x)


class ResNet152(nn.Module):

    def __init__(self,):
        super(ResNet152,self).__init__()

        self.resnet = models.resnet152(weights="ResNet152_Weights.IMAGENET1K_V2")
        self.in_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
                nn.Linear(self.in_ftrs, nclasses)
                )

        ct = 0
        for child in self.resnet.children():
            ct += 1
            if ct < 7:
                for param in child.parameters():
                    param.requires_grad = False
        # This freezes layers 1-6 in the total 10 layers of Resnet101
                
    def forward(self,x):
        return self.resnet(x)
    
    
class InceptionV3(nn.Module):

    def __init__(self,):
        super(InceptionV3,self).__init__()

        self.incep = models.inception_v3(weights= "Inception_V3_Weights.DEFAULT")
        self.in_ftrs = self.incep.fc.in_features
        self.incep.fc = nn.Sequential(
                nn.Linear(self.in_ftrs, nclasses)
               )
        
        ct = 0
        for child in self.incep.children():
            ct += 1
            if ct < 17:
                for param in child.parameters():
                    param.requires_grad = False
        #This freezes layers 1-16 in the total 22 layers of inceptionV3
        
                    
        # check            
        # for name, child in self.incep.named_children():
        #     print(name)
        #     i=0
        #     for param in child.parameters():
        #         if i ==0 :
        #             print(param.requires_grad)
        #         i+=1
                
    def forward(self,x):
        return self.incep(x)
    


class vit_l_16_1layer(nn.Module):

    def __init__(self,):
        super(vit_l_16_1layer,self).__init__()
        self.vitnet = models.vit_l_16(weights="ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1")
        self.out_ftrs = self.vitnet.heads.head.out_features
        
        for param in self.vitnet.parameters() :
            param.requires_grad=False
        
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
        
        self.fc = nn.Linear(self.out_ftrs,nclasses)
        
        #self.vitnet.heads.fc = nn.Sequential(
        #    self.dropout,
        #    nn.Linear(self.out_ftrs,nclasses)
        #    )
            
        
    def forward(self,x):
        x = self.vitnet(x)
        x=self.dropout(x)

        return self.fc(x)

    
class vit_l_16_2layers(nn.Module):

    def __init__(self,):
        super(vit_l_16_2layers,self).__init__()
        self.vitnet = models.vit_l_16(weights="ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1")
        self.out_ftrs = self.vitnet.heads.head.out_features
        
        

        for param in self.vitnet.parameters() :
            param.requires_grad=False
        
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()

        #old
        self.fc1 = nn.Linear(self.out_ftrs,320)
        self.fc2 = nn.Linear(320, nclasses)

        #new
        #self.vitnet.heads.fc = nn.Sequential(
        #    self.dropout,
        #    nn.Linear(self.out_ftrs,320)
        #    )
        
        #self.fc = nn.Linear(320, nclasses)

    def forward(self,x):
        x = self.vitnet(x)
        
        #old
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        
        #new
        #x = F.relu(self.dropout(x))
        #return self.sigmoid(self.fc(x))
        


class vit_l_16_3layers(nn.Module):

    def __init__(self,):
        super(vit_l_16_3layers,self).__init__()
        self.vitnet = models.vit_l_16(weights="ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1")
        self.out_ftrs = self.vitnet.heads.head.out_features
        
        

        for param in self.vitnet.parameters() :
            param.requires_grad=False
        
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
    
        #self.vitnet.heads.fc = nn.Sequential(
        #    self.dropout,
        #    nn.Linear(self.out_ftrs,520)
        #    )
        
        self.fc1 = nn.Linear(self.out_ftrs,520)
        self.fc2 = nn.Linear(520,210)
        self.fc3 = nn.Linear(210, nclasses)
        
    def forward(self,x):
        x = self.vitnet(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)
        


class vit_l_16_4layers(nn.Module):

    def __init__(self,):
        super(vit_l_16_4layers,self).__init__()
        self.vitnet = models.vit_l_16(weights="ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1")
        self.out_ftrs = self.vitnet.heads.head.out_features
        
        

        for param in self.vitnet.parameters() :
            param.requires_grad=False
        
        self.dropout = nn.Dropout(0.25)
        self.sigmoid = nn.Sigmoid()
    
        #self.vitnet.heads.fc = nn.Sequential(
        #    self.dropout,
        #    nn.Linear(self.out_ftrs,520)
        #    )
        
        self.fc1 = nn.Linear(self.out_ftrs,520)
        self.fc2 = nn.Linear(520,210)
        self.fc3 = nn.Linear(210, 100)
        self.fc4 = nn.Linear(100, nclasses)
        
    def forward(self,x):
        x = self.vitnet(x)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class vit_h_14(nn.Module):

    def __init__(self,):
        super(vit_h_14,self).__init__()
        self.vitnet = models.vit_h_14(weights="ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1")
        self.out_ftrs = self.vitnet.heads.head.out_features
        
        

        for param in self.vitnet.parameters() :
            param.requires_grad=False
        
        # self.fc1 = nn.Linear(320, 50)
        self.fc = nn.Linear(self.out_ftrs, nclasses)
        
        
    def forward(self,x):
        x = self.vitnet(x)
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        return self.fc(x)





class regnet_y_32gf(nn.Module):

    def __init__(self,):
        super(regnet_y_32gf,self).__init__()
        self.regnet = models.regnet_y_32gf(weights="RegNet_Y_32GF_Weights.IMAGENET1K_SWAG_E2E_V1")
        self.out_ftrs = self.regnet.fc.out_features
        
        

        for param in self.regnet.parameters() :
            param.requires_grad=False
        
        # self.fc1 = nn.Linear(320, 50)
        self.fc = nn.Linear(self.out_ftrs, nclasses)
        
        
    def forward(self,x):
        x = self.regnet(x)
        #x = x.view(-1, 320)
        #x = F.relu(self.fc1(x))
        return self.fc(x)