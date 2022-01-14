import torch.utils.data as data
import torch
import os
import cv2 as cv
cv.setNumThreads(0) 
import numpy as np
import config
import time
from utils import ImgRead,ImgWrite, ToneSimple,ImgReadWithPrefix,ReadData

class MedTrainDataset(data.Dataset):
    def __init__(self, typeIndicator, transform=None):  
        self.indicator = typeIndicator
        
        self.totalNum = 0

        
        self.imgSet = []
        for path in config.basePaths:
            imgs = os.listdir(path)
            setNum = len(imgs)
            self.imgSet.append(imgs)
            self.totalNum += setNum

    def mapIndex2PathAndIndex(self, index):
        remain = index
        for setIndex,ims in enumerate(self.imgSet):
            if remain < len(ims):
                return config.basePaths[setIndex], ims[remain].split(".")[1]
            else:
                remain -= len(ims)

        return None, -1
    def __getitem__(self, index):
        path, idx = self.mapIndex2PathAndIndex(index)
        try:
            img, img_2, img_3, occ_warp_img, woCheckimg, woCheckimg_2, woCheckimg_3, labelimg, metalic, Roughnessimg, Depthimg, Normalimg = ReadData(path+"compressed.{}.npz".format(idx))
        except:
            print(path)
            print(idx)

        input = img
        mask = input.copy()
        mask[mask==0.]=1.0
        mask[mask==-1]=0.0
        mask[mask!=0.0]=1.0

        # occ_warp_img[mask!=0.0] = 0.0
        occ_warp_img[occ_warp_img < 0.0] = 0.0
        woCheckimg[woCheckimg < 0.0] = 0.0
        woCheckimg_2[woCheckimg_2 < 0.0] = 0.0
        woCheckimg_3[woCheckimg_3 < 0.0] = 0.0



        labelimg = ToneSimple(labelimg)
        occ_warp_img = ToneSimple(occ_warp_img)
        woCheckimg = ToneSimple(woCheckimg)


        features = np.concatenate([Normalimg,Depthimg,Roughnessimg,metalic], axis=2)

        
        mask2 = img_2.copy()
        mask2[mask2==0.]=1.0
        mask2[mask2==-1]=0.0
        mask2[mask2!=0.0]=1.0

        mask3 = img_3.copy()
        mask3[mask3==0.]=1.0
        mask3[mask3==-1]=0.0
        mask3[mask3!=0.0]=1.0

        woCheckimg_2 = ToneSimple(woCheckimg_2)

        woCheckimg_3 = ToneSimple(woCheckimg_3)

        finalMask = np.repeat(mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1)), 6, axis=2)

        his_1 = np.concatenate([woCheckimg, mask[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_2 = np.concatenate([woCheckimg_2, mask2[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])
        his_3 = np.concatenate([woCheckimg_3, mask3[:,:,0].reshape((Normalimg.shape[0],Normalimg.shape[1], 1))], axis=2).transpose([2,0,1]).reshape(1, 4, Normalimg.shape[0],Normalimg.shape[1])

        input = np.concatenate([woCheckimg,occ_warp_img],axis=2)
        

        hisBuffer = np.concatenate([his_3, his_2, his_1], axis=0)

        return torch.tensor(input.transpose([2,0,1])),torch.tensor(features.transpose([2,0,1])), torch.tensor(finalMask.transpose([2,0,1])), torch.tensor(hisBuffer), torch.tensor(labelimg.transpose([2,0,1]))

    def __len__(self):
        return self.totalNum

