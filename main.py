import argparse
from time import gmtime, strftime
import os
import torch
from torch.utils.data import DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder

from arch.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import train
import val
# Flexible integration for any Python script
import wandb
import numpy as np
from Mdataloader import classification
# 1. Start a W&B run
wandb.init(project='proto', entity='minwoo1')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50'])
    parser.add_argument('--lr_base', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr_drop_epochs', type=int, default=[15,40,60], nargs='+')
    parser.add_argument('--lr_drop_rate', type=float, default=0.04)
    args = parser.parse_args()
    wandb.config.update(args)
         #wandb
  
    # define model
    if args.arch.startswith('resnet'):
        if args.arch == 'resnet18':
            model = resnet18(num_classes=4)
        elif args.arch == 'resnet34':
            model = resnet34(num_classes=4)
        elif args.arch == 'resnet50':
            model = resnet50(num_classes=4)
        elif args.arch == 'resnet101':
            model = resnet101(num_classes=4)
        elif args.arch == 'resnet152':
            model = resnet152(num_classes=4)
        else:
            raise NotImplementedError(f"architecture {args.arch} is not implemented")
    else:
        raise NotImplementedError(f"architecture {args.arch} is not implemented")
    model = model.cuda()
    model = torch.nn.parallel.DataParallel(model)
    wandb.watch(model)

#    transform_train = transforms.Compose([transforms.Resize((64,64)),transforms.RandomVerticalFlip(p=1),transforms.ToTensor()]) 
 #   transform_val = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()]) 
  #  dataloader_train = DataLoader(classification(1, transform_train), shuffle=True, num_workers=10, batch_size=args.batch_size)
   # dataloader_val = DataLoader(classification(0, transform_val), shuffle=False, num_workers=10, batch_size=args.batch_size)
    transform_train = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()]) 
    transform_val = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor()]) 
    dataloader_train = DataLoader(classification(1, transform_train), shuffle=True, num_workers=10, batch_size=args.batch_size)
    dataloader_val = DataLoader(classification(0, transform_val), shuffle=False, num_workers=10, batch_size=args.batch_size)
    

    train_meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in dataloader_train]
    train_stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in dataloader_train]
 
    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])    
    print(f"train_meanR : {train_meanR}  , train_meanG : {train_meanG}, train_meanB : {train_meanB}")
    print(f"train_stdR : {train_stdR} , train_stdG : {train_stdR}, train_stdB : {train_stdR} ")


    
    transform_train = transforms.Compose([transforms.Resize((64,64)),transforms.RandomVerticalFlip(p=1),transforms.ToTensor(), transforms.Normalize((train_meanR,train_meanG,train_meanB),(train_stdR,train_stdG,train_stdB))])
    transform_val = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(), transforms.Normalize((train_meanR,train_meanG,train_meanB),(train_stdR,train_stdG,train_stdB))])
    dataloader_train = DataLoader(classification(1, transform_train), shuffle=True, num_workers=10, batch_size=args.batch_size)
    dataloader_val = DataLoader(classification(0, transform_val), shuffle=False, num_workers=10, batch_size=args.batch_size)
    

  
    # LR schedule
    lr = args.lr_base
    lr_per_epoch = []
    for epoch in range(args.epochs):
        if epoch in args.lr_drop_epochs:
            lr *= args.lr_drop_rate
        lr_per_epoch.append(lr)

    # define loss and optimizer
    criterion = torch.nn.CrossEntropyLoss().cuda()
#    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, momentum=0.9, weight_decay=5e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_base, weight_decay=5e-4)
    # save_path
    current_time = strftime('%Y-%m-%d_%H:%M', gmtime())
    save_dir = os.path.join(f'checkpoints/{current_time}')
    os.makedirs(save_dir,  exist_ok=True)

    # train and val
    best_perform, best_epoch = -100, -100
    for epoch in range(1, args.epochs+1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_per_epoch[epoch-1]
 

        train.train(model, dataloader_train, criterion, optimizer, epoch=epoch)
        acc1, acc4 = val.val(model, dataloader_val, epoch=epoch)

        save_data = {'epoch': epoch,
                     'acc1': acc1,
                     'acc4': acc4,
                     'state_dict': model.state_dict(),
                     'optimizer': optimizer.state_dict()}
        
        torch.save(save_data, os.path.join(save_dir, f'{epoch:03d}.pth.tar'))
        if epoch > 1:
            os.remove(os.path.join(save_dir, f'{epoch-1:03d}.pth.tar'))
        if acc1 >= best_perform:
            torch.save(save_data, os.path.join(save_dir, 'best.pth.tar'))
            best_perform = acc1
            best_epoch = epoch
        print(f"best performance {best_perform} at epoch {best_epoch}")
        wandb.log({"acc_val":acc1})

wandb.finish()
