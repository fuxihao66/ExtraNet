import torch
import os
mSkip=True
mLossHoleArgument=1
mLossHardArgument=1
p="C:/Users/admin/"
basePaths = [p+"Medieval/Medieval_0/"]  ## Path to all compressed data set folders


TestMetalicPrefix = "Metallic"

NormalPrefix="WorldNormal"
DepthPrefix="SceneDepth"
RoughnessPrefix="Roughness"

warpPrefix = "Warp"

ofwarpPrefix = "WarpDiffuse"


TestNormalPrefix="WorldNormal"
TestDepthPrefix="SceneDepth"
TestRoughnessPrefix="Roughness"
mdevice=torch.device("cuda:0")

#Training related
learningrate=1e-3
epoch=100
printevery=50
batch_size=8
