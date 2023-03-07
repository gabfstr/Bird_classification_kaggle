import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
from torch.autograd import Variable
from tqdm import tqdm
from datetime import datetime
import csv 

# Training settings
parser = argparse.ArgumentParser(description='RecVis A3 training script')
parser.add_argument('--data', type=str, default='bird_dataset', metavar='D',
                    help="folder where data is located. train_images/ and val_images/ need to be found in the folder")
parser.add_argument('--batch-size', type=int, default=64, metavar='B',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=15, metavar='N',
                    help='number of epochs to train (default: 15)')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--experiment', type=str, default='experiment', metavar='E',
                    help='folder where experiment outputs are located.')
parser.add_argument('--resume-training', type=str, default='', metavar='C',
                    help='model selected to resume training.')

parser.add_argument('--n-layers', type=int, default='1', metavar='NL',
                    help='if vit model, number of learning layers on top')

args = parser.parse_args()
#use_mps = torch.backends.mps.is_available()
#use_cuda = torch.cuda.is_available() * (1-use_mps)

#if use_mps : 
#    device = torch.device("mps")
use_cuda = torch.cuda.is_available()
#elif use_cuda:
if use_cuda : 
    device = torch.device("cuda")
else :
    device = torch.device("cpu")
torch.manual_seed(args.seed)


def main():
    # Create experiment folder
    if not os.path.isdir(args.experiment):
        os.makedirs(args.experiment)

    # Data initialization and loading
    from data import data_transforms

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/train_images',
                            transform=data_transforms["train"]),
        batch_size=args.batch_size, shuffle=True, num_workers=1)
    

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.data + '/val_images',
                            transform=data_transforms["val"]),
        batch_size=args.batch_size, shuffle=False, num_workers=1)
    
    # Neural network and optimizer
    # We define neural net in model.py so that it can be reused by the evaluate.py script
    from model import vit_l_16_1layer, vit_l_16_2layers, vit_l_16_3layers, vit_l_16_4layers
    
    if args.resume_training != '':
        try :
            state_dict = torch.load(args.resume_training)
            print("\nResuming training of model",args.resume_training)
            model = vit_l_16().to(device)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print("\nNo corresponding model found")
            model = vit_l_16().to(device)
    else : 
        if(args.n_layers)==2:
            model = vit_l_16_2layers().to(device)
        elif(args.n_layers)==3:
            model = vit_l_16_3layers().to(device)
        elif(args.n_layers)==4:
            model = vit_l_16_4layers().to(device)
        else :
            model = vit_l_16_1layer().to(device)
        
        print("\nTraining {} model with {} layers, learning rate {}, momentum {} and batch size {}\n".format(
            model.__class__.__name__,args.n_layers, args.lr, args.momentum, args.batch_size))
    #if use_mps:
    #    print('Using MPS')
    #    #model.to(device)

    # elif use_cuda:
    if use_cuda:
        print('Using GPU\n')
        #model.cuda()
    else:
        print('Using CPU\n')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    #optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9,0.999), eps=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,verbose=True,patience=4, min_lr=1e-5)

    def train(epoch):
        
        model.train()
        correct=0
        total_loss=0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # if use_cuda:
            #     data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
            #loss.requires_grad = True
            loss.backward()
            optimizer.step()
            total_loss+=loss.data.item()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        total_loss=total_loss/len(train_loader.dataset)
        train_acc = (100. * correct / len(train_loader.dataset)).data.item()
        return total_loss, train_acc
        

    def validation():
        model.eval()
        validation_loss = 0
        correct = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            #if use_cuda:
            #    data, target = data.cuda(), target.cuda()
            output = model(data)
            # sum up batch loss
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            validation_loss += criterion(output, target).data.item()
            # get the index of the max log-probability
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

        validation_loss /= len(val_loader.dataset)
        validation_acc = (100. * correct / len(val_loader.dataset)).data.item()
        print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
            validation_loss, correct, len(val_loader.dataset),
            validation_acc))
        return validation_loss, validation_acc
    
    rows=[]
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train(epoch)
        val_loss, val_acc = validation()
        scheduler.step(val_loss)
        #print("train_loss : {:.6f} val_loss : {:.6f}\ntrain_acc : {:.1f}% val_acc : {:.1f}%".format(
        #        train_loss,val_loss, train_acc, val_acc))
        
        #storing results for csv
        rows.append([epoch,train_loss,val_loss, train_acc, val_acc])
        
        
        #if epoch > 15:
        if val_loss < 0.0085 or (round(optimizer.param_groups[0]['lr'],6) == scheduler.min_lrs[0]) or epoch == 50:
            model_file = args.experiment + '/model_' + str(epoch) +"_"+str(args.n_layers)+"layers" + '.pth'
            torch.save(model.state_dict(), model_file)
            print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file')
        print()
        
        #print("learning rate :",optimizer.param_groups[0]['lr'])
        if(round(optimizer.param_groups[0]['lr'],6) == scheduler.min_lrs[0]):
            print("\nConvergence reached")
            return
        
    #writing results in csv
    fields = ["epoch", "train_loss","val_loss", "train_acc", "val_acc"]
    filename = "results_test.csv"
    with open(filename, 'w') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile)   
        # writing the fields 
        csvwriter.writerow(fields)  
        # writing the data rows 
        csvwriter.writerows(rows)


if __name__ == '__main__':
    start = datetime.now()
    main()
    end=datetime.now()
    exec_time = (end - start).total_seconds()
    print("\n")
    print("Total running time : {0} min {1} s".format( int(exec_time // 60), round(exec_time % 60),0))