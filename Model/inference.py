import Loaders
import ExtraNet
import torch.nn as nn
import config
import cv2 as cv
cv.setNumThreads(0) 
import glob
import numpy as np
from utils import ImgRead,ImgWrite, ToneSimple, DeToneSimple,ImgReadWithPrefix, ReadData
import torch.utils.data as data
import torch.nn.functional as F
from torchvision.utils import save_image
from torch import optim
import Losses
import os
import torch
import time
from torch.utils.tensorboard import SummaryWriter



def GetStartEnd(path):
    start = 99999
    end = 0
    prefix = None
    for filePath in glob.glob(path + "GT/*"):
        if prefix == None:
            prefix = filePath.split("\\")[-1].split('.')[0][:-2]
        idx = int(filePath.split('.')[1])
        start = min(start, idx)
        end = max(end, idx)
    
    print(prefix)
    return start, end+1, prefix


def inference(modelPath):

    model = ExtraNet.ExtraNet(18,3)


    model_CKPT = torch.load(modelPath, map_location="cuda:0")
    model.load_state_dict(model_CKPT['state_dict'])
    model=model.cuda()
    model.eval()
    with torch.no_grad():
        path = "../TestData/"
        prefix = "MedievalDocks"
        idx = 339


        img = ImgReadWithPrefix(path+"warp_res", int(idx),"1", cvtrgb=True)
        img_2 = ImgReadWithPrefix(path+"warp_res", int(idx),"3", cvtrgb=True)
        img_3 = ImgReadWithPrefix(path+"warp_res", int(idx),"5", cvtrgb=True)


        warp_image = ImgReadWithPrefix(path+"warp_no_hole",idx,"1",prefix=prefix+config.warpPrefix,cvtrgb=True)
        warp_image_2 = ImgReadWithPrefix(path+"warp_no_hole",idx,"3",prefix=prefix+config.warpPrefix,cvtrgb=True)
        warp_image_3 = ImgReadWithPrefix(path+"warp_no_hole",idx,"5",prefix=prefix+config.warpPrefix,cvtrgb=True)
    

        Normalimg = ImgRead(path, idx, prefix=prefix+config.TestNormalPrefix, cvtrgb=True)
        metalicimg = ImgRead(path, idx, prefix=prefix+config.TestMetalicPrefix, cvtrgb=True)[:,:,0:1]
        Depthimg = ImgRead(path, idx, prefix=prefix+config.TestDepthPrefix, cvtrgb=True)[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))
        Depthimg[Depthimg > 100] = 0.0
        Depthimg = (Depthimg - Depthimg.min()) / (Depthimg.max() - Depthimg.min() + 1e-6)
        Roughnessimg = ImgRead(path, idx, prefix=prefix+config.TestRoughnessPrefix, cvtrgb=True)[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))
        
        

        occ_warp_img = ImgReadWithPrefix(path+"occ",idx,"1",prefix=prefix+config.warpPrefix, cvtrgb=True)

        input = img
        mask = input.copy()
        mask[mask==0.]=1.0
        mask[mask==-1]=0.0
        mask[mask!=0.0]=1.0

        warp_image[warp_image < 0.0] = 0.0
        occ_warp_img[occ_warp_img < 0.0] = 0.0
        warp_image_2[warp_image_2 < 0.0] = 0.0
        warp_image_3[warp_image_3 < 0.0] = 0.0



        occ_warp_img = ToneSimple(occ_warp_img)
        warp_image = ToneSimple(warp_image)

        features = np.concatenate([Normalimg,Depthimg,Roughnessimg,metalicimg], axis=2)


        input = np.concatenate([warp_image, occ_warp_img],axis=2).transpose([2,0,1])


        mask2 = img_2.copy()
        mask2[mask2==0.]=1.0
        mask2[mask2==-1]=0.0
        mask2[mask2!=0.0]=1.0

        mask3 = img_3.copy()
        mask3[mask3==0.]=1.0
        mask3[mask3==-1]=0.0
        mask3[mask3!=0.0]=1.0

        warp_image_2 = ToneSimple(warp_image_2)

        
        warp_image_3 = ToneSimple(warp_image_3)

        his_1 = np.concatenate([warp_image, mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_2 = np.concatenate([warp_image_2, mask2[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_3 = np.concatenate([warp_image_3, mask3[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])

        hisBuffer = np.concatenate([his_3, his_2, his_1], axis=0)   


        mask = np.repeat(mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1)), 6, axis=2).transpose([2,0,1])

        features = features.transpose([2,0,1])

        hisBuffer = torch.tensor(hisBuffer).unsqueeze(0).cuda()
        input = torch.tensor(input).unsqueeze(0).cuda()
        features = torch.tensor(features).unsqueeze(0).cuda()
        mask=torch.tensor(mask).unsqueeze(0).cuda()

        res=model(input, features, mask, hisBuffer)


        res=res.squeeze(0).cpu().numpy().transpose([1,2,0])
        res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
        

        res = DeToneSimple(res)


        albedo = ImgRead(path, idx,prefix=prefix+"BaseColor", cvtrgb=False)

        skybox = ImgRead(path, idx,prefix=prefix+"Skybox", cvtrgb=False)
        # skybox = ImgRead(path, idx,prefix=prefix+"PreTonemapHDRColor", cvtrgb=False)  # For simplicity, the skybox can be extracted from PreTonemapHDRColor file by counting only pixels with normal of (-1,-1,-1).


        res=res*albedo

        hole = np.logical_and(Normalimg[:,:,0] == -1, Normalimg[:,:,1] == -1)
        hole = np.logical_and(Normalimg[:,:,2] == -1, hole)
        res[hole] = skybox[hole]
        ImgWrite("../TestData","res",idx,res)
                

def Tensor2NP(t):
    res=t.squeeze(0).cpu().numpy().transpose([1,2,0])
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    res = DeToneSimple(res)

    return res


if __name__ =="__main__":
    inference('medieval.pth.tar')
    
    


