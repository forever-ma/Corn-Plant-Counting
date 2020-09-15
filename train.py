import numpy as np
import time
import torch
import torch.nn as nn
import os
import visdom
import random
from tqdm import tqdm as tqdm

from network import CORNNet
from my_dataset import CornDataset
from loss import My_loss

if __name__=="__main__":
    train_image_root='/home/user/corn/data/train/images'
    train_dmap_root='/home/user/corn/data/train/ground_truth'
    test_image_root='/home/user/corn/data/test/images'
    test_dmap_root='/home/user/corn/data/test/ground_truth'
    gpu_or_cpu='cuda' 
    lr                = 1e-7
    batch_size        = 1
    momentum          = 0.95
    epochs            = 3000
    steps             = [-1,1,100,150]
    scales            = [1,1,1,1]
    workers           = 4
    seed              = time.time()
    print_freq        = 30 
    
    vis=visdom.Visdom()
    device=torch.device(gpu_or_cpu)
    torch.cuda.manual_seed(seed)
    model=CORNNet().to(device)
    criterion1=nn.MSELoss(size_average=False).to(device)
    criterion2=My_loss().to(device)
 #   optimizer=torch.optim.SGD(model.parameters(),lr,
 #                             momentum=momentum,
 #                             weight_decay=0)
    optimizer=torch.optim.Adam(model.parameters(),lr,
                              betas=(0.9,0.99))
    train_dataset=CornDataset(train_image_root,train_dmap_root,gt_downsample=8,phase='train')
    train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=1,shuffle=True)
    test_dataset=CornDataset(test_image_root,test_dmap_root,gt_downsample=8,phase='test')
    test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=1,shuffle=False)
    
    if not os.path.exists('./checkpoints'):
        os.mkdir('./checkpoints')
    min_mae=10000
    min_mse=10000
    min_epoch=0
    train_loss_list=[]
    epoch_list=[]
    test_error_list=[]
    for epoch in range(0,epochs):
        # training phase
        model.train()
        epoch_loss=0
        for i,(img,gt_dmap) in enumerate(tqdm(train_loader)): 
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            et_dmap = model(img)
            loss1=criterion1(et_dmap,gt_dmap)
            loss2=criterion2(et_dmap.data.sum(), gt_dmap.data.sum())
            loss=0.9*loss1+0.1*loss2
            epoch_loss+=loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("epoch:",epoch,"loss:",epoch_loss/len(train_loader))
        epoch_list.append(epoch)
        train_loss_list.append(epoch_loss/len(train_loader))
        torch.save(model.state_dict(),'./checkpoints/epoch_'+str(epoch)+".pth")
    
        # testing phase
        model.eval()
        with torch.no_grad():
            mae = 0
            mse = 0
            for i, (img, gt_dmap) in enumerate(tqdm(test_loader)):  
                img = img.to(device)
                gt_dmap = gt_dmap.to(device)
                et_dmap = model(img)
                mae += abs(et_dmap.data.sum() - gt_dmap.data.sum()).item()
                mse += ((et_dmap.data.sum()-gt_dmap.data.sum())*(et_dmap.data.sum()-gt_dmap.data.sum())).item()
                del img, gt_dmap, et_dmap
                
            test_error_list.append(mae / len(test_loader))
            print("epoch:"+str(epoch)+" error:"+str(mae/len(test_loader))+" mae:"+str(mae/len(test_loader))+" mse:"+str(np.sqrt(mse/len(test_loader))))

        vis.line(win=1,X=epoch_list, Y=train_loss_list, opts=dict(title='train_loss'))
        vis.line(win=2,X=epoch_list, Y=test_error_list, opts=dict(title='test_error'))
        # show an image
        index=random.randint(0,len(test_loader)-1)
        img,gt_dmap=test_dataset[index]
        vis.image(win=3,img=img,opts=dict(title='img'))
        vis.image(win=4,img=gt_dmap/(gt_dmap.max())*255,opts=dict(title='gt_dmap('+str(gt_dmap.sum())+')'))
        img=img.unsqueeze(0).to(device)
        gt_dmap=gt_dmap.unsqueeze(0)
        et_dmap=model(img)
        et_dmap=et_dmap.squeeze(0).detach().cpu().numpy()
        vis.image(win=5,img=et_dmap/(et_dmap.max())*255,opts=dict(title='et_dmap('+str(et_dmap.sum())+')'))

    import time
    print(time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(time.time())))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
