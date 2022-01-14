import Loaders
import ExtraNet
import torch.nn as nn
import config
import cv2 as cv
cv.setNumThreads(0) 
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
#from torch.utils.tensorboard import SummaryWriter



def Tensor2NP(t):
    res=t.squeeze(0).cpu().numpy().transpose([1,2,0])
    res=cv.cvtColor(res,cv.COLOR_RGB2BGR)
    res = DeToneSimple(res)

    return res



def train(dataLoaderIns, modelSavePath):
    model = ExtraNet.ExtraNet(18,3)

    model = model.to(config.mdevice)
    optimizer = optim.Adam(model.parameters(), lr=config.learningrate)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config.epoch,eta_min=1e-6)

    criterion = Losses.mLoss()

    for e in range(config.epoch):
        model.train()

        iter=0
        loss_all=0
        startTime = time.time()

        for input,features,mask,hisBuffer,label in dataLoaderIns:
            input=input.cuda()
            hisBuffer=hisBuffer.cuda()
            mask=mask.cuda()
            features=features.cuda()
            label=label.cuda()

            optimizer.zero_grad()


            res=model(input,features, mask, hisBuffer)
            
            loss=criterion(res,mask,label)
            
            loss.backward()
            optimizer.step()
            iter+=1
            loss_all+=loss
            if iter%config.printevery==1:
                print(loss)
        
        endTime = time.time()
        print("epoch time is {}".format(endTime - startTime))

        
        print("epoch %d mean loss for train is %f"%(e,loss_all/iter))

        if e > 20:
            scheduler.step()

        if e % 5 == 0:
            torch.save({'epoch': e + 1, 'state_dict': model.state_dict(), 
                        'optimizer': optimizer.state_dict()},
                         './totalModel.{}.pth.tar'.format(e))

    torch.save(model.state_dict(), modelSavePath)

    
    
    


 
if __name__ =="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    trainDiffuseDataset = Loaders.MedTrainDataset(0)
    trainDiffuseLoader = data.DataLoader(trainDiffuseDataset,config.batch_size,shuffle=True,num_workers=6, pin_memory=True)
    
    train(trainDiffuseLoader, "./finalModel.pkl")
   
