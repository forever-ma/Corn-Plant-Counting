import torch
import matplotlib.pyplot as plt
import matplotlib.cm as CM
from tqdm import tqdm
import numpy as np

from network import CORNNet
from my_dataset import CornDataset


def cal_mae(img_root,gt_dmap_root,model_param_path):
    '''
    Calculate the MAE of the test data.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific model parameters.
    '''
    device=torch.device("cuda")
    model=CORNNet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    model.to(device)
    dataset=CornDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    mae=0
    mse=0
    with torch.no_grad():
        for i,(img,gt_dmap) in enumerate(tqdm(dataloader)):
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            # forward propagation
            et_dmap=model(img)
            mae+=abs(et_dmap.data.sum()-gt_dmap.data.sum()).item()
            mse+=((et_dmap.data.sum() - gt_dmap.data.sum()) * (et_dmap.data.sum() - gt_dmap.data.sum())).item()
            del img,gt_dmap,et_dmap

    print("model_param_path:"+model_param_path+" mae:"+str(mae/len(dataloader))+" mse:"+str(np.sqrt(mse/len(dataloader))))

def estimate_density_map(img_root,gt_dmap_root,model_param_path,index):
    '''
    Show one estimated density-map.
    img_root: the root of test image data.
    gt_dmap_root: the root of test ground truth density-map data.
    model_param_path: the path of specific model parameters.
    index: the order of the test image in test dataset.
    '''
    device=torch.device("cuda")
    model=CORNNet().to(device)
    model.load_state_dict(torch.load(model_param_path))
    dataset=CornDataset(img_root,gt_dmap_root,8,phase='test')
    dataloader=torch.utils.data.DataLoader(dataset,batch_size=1,shuffle=False)
    model.eval()
    for i,(img,gt_dmap) in enumerate(dataloader):
        if i==index:
            img=img.to(device)
            gt_dmap=gt_dmap.to(device)
            et_dmap=model(img).detach()
            et_dmap=et_dmap.squeeze(0).squeeze(0).cpu().numpy()
            print(et_dmap.shape)
            plt.imshow(et_dmap,cmap=CM.jet)
            break


if __name__=="__main__":
    torch.backends.cudnn.enabled=False
    img_root='/home/user/corn/data/test/images'  
    gt_dmap_root='/home/user/corn/data/test/ground_truth'   
    model_param_path='/home/user/corn/checkpoints/epoch_580.pth'  
    cal_mae(img_root,gt_dmap_root,model_param_path)
    # estimate_density_map(img_root,gt_dmap_root,model_param_path,3) 