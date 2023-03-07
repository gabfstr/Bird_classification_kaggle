import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Net, ResNet50, ResNet101, ResNet50Attention, vit_l_16_1layer,vit_l_16_2layers,vit_l_16_3layers,vit_l_16_4layers

parser = argparse.ArgumentParser(description='RecVis A3 evaluation script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. test_images/ need to be found in the folder")
parser.add_argument('--model', default = '', type=str, metavar='M',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--models', nargs='*', type=str, default = '', metavar='MS',
                    help="the model file to be evaluated. Usually it is of the form model_X.pth")
parser.add_argument('--outfile', type=str, default='experiment/kaggle.csv', metavar='D',
                    help="name of the output csv file")

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

single_model = args.models==''

if single_model:
    state_dict = torch.load(args.model)
    print("\nEvaluating model", args.model,"\n")
    model = vit_l_16_4layers()

else :
    print("\nEvaluating models", args.models,"with horizontal voting procedure\n".format())
    state_dicts = [torch.load(x) for x in args.models]

    model1 = vit_l_16_2layers()
    model2= vit_l_16_3layers()
    model3= vit_l_16_4layers()

if use_cuda:
    print('Using GPU\n')
    device = torch.device("cuda")
    if single_model :
        model.cuda()
    else :
        model1.cuda()
        model2.cuda()
        model3.cuda()
else:
    print('Using CPU\n')
    device = torch.device("cpu")

from data import data_transforms

test_dir = args.data + '/test_images/mistery_category'

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


output_file = open(args.outfile, "w")
output_file.write("Id,Category\n")
for f in tqdm(os.listdir(test_dir)):
    if 'jpg' in f:
        data = data_transforms["test"](pil_loader(test_dir + '/' + f))
        data = data.view(1, data.size(0), data.size(1), data.size(2))
        if use_cuda:
            data = data.cuda()
        if single_model :
            model.load_state_dict(state_dict)
            model.eval()
            output = model(data)
        else : 

            output=torch.zeros(size=(1,20)).to(device)
            i=0
            for instance in state_dicts:
                if i ==0 :
                    model1.load_state_dict(instance)
                    model1.eval()
                    output1 = F.sigmoid(model1(data))
                    output += output1
                elif i ==1 : 
                    model2.load_state_dict(instance)
                    model2.eval()
                    output2 = F.sigmoid(model2(data))
                    output += output2
                elif i==2:
                    model3.load_state_dict(instance)
                    model3.eval()
                    output3 = F.sigmoid(model3(data))
                    output += model3(data)
                i+=1
                
                
                #model.load_state_dict(instance)
                #model.eval()
                #output += F.sigmoid(model(data))
                
                #output1=model(data)
                #print("output :",output,"\n\n")
        pred = output.data.max(1, keepdim=True)[1]
        #pred1 = output1.data.max(1,keepdim=True)[1]
        #pred2 = output1.data.max(1,keepdim=True)[1]
        #print("pred1==pred2 :",(pred2==pred1))
        output_file.write("%s,%d\n" % (f[:-4], pred))

output_file.close()

print("Succesfully wrote " + args.outfile + ', you can upload this file to the kaggle competition website')
        


