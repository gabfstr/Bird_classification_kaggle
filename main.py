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
args = parser.parse_args()
use_cuda = torch.cuda.is_available()
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
    from model import Net
    model = Net()

    if use_cuda:
        print('Using GPU')
        model.cuda()
    else:
        print('Using CPU')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


    def train(epoch):
        
        print(model,"\n")

        model.train()
        correct=0
        total_loss=0
        for batch_idx, (data, target) in enumerate(train_loader):
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            criterion = torch.nn.CrossEntropyLoss(reduction='mean')
            loss = criterion(output, target)
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
            if use_cuda:
                data, target = data.cuda(), target.cuda()
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
        print("train_loss : {:.6f} val_loss : {:.6f}\ntrain_acc : {:.1f}% val_acc : {:.1f}%".format(
                train_loss,val_loss, train_acc, val_acc))
        #storing results for csv
        rows.append([epoch,train_loss,val_loss, train_acc, val_acc])


        model_file = args.experiment + '/model_' + str(epoch) + '.pth'
        torch.save(model.state_dict(), model_file)
        print('Saved model to ' + model_file + '. You can run `python evaluate.py --model ' + model_file + '` to generate the Kaggle formatted csv file\n')
    
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