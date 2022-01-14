import torch
import numpy as np
import torch.nn as nn
from config import mLossHoleArgument,mLossHardArgument


class LossHoleArgument(nn.Module):
    def __init__(self):
        super(LossHoleArgument,self).__init__()
    def forward(self,input,mask,target):
        lossMask=1-mask[:,:3,:,:]
        lholeAugment=(lossMask*torch.abs(input-target)).sum()/lossMask.sum()
        return lholeAugment
class LossHardArgument(nn.Module):
    def __init__(self,ratio=0.1):
        super(LossHardArgument,self).__init__()
        self.ratio=ratio
    def forward(self,input,target):
        n,c,h,w = input.shape
        val,ind=torch.topk(torch.abs(input-target).view(n,c,-1),k=int(h*w*self.ratio))
        return val.mean()
class mLoss(nn.Module):
    def __init__(self):
        super(mLoss,self).__init__()
        self.hole=LossHoleArgument()
        self.hard=LossHardArgument()
    def forward(self,input,mask,target):
        basicl1=torch.abs(input-target).mean()
        if mLossHoleArgument:
            basicl1+=self.hole(input,mask,target)*mLossHoleArgument
        if mLossHardArgument:
            basicl1+=self.hard(input,target)*mLossHardArgument
        return basicl1

