import os 
import numpy as np
import cv2
from math import ceil,floor
from math import exp
from math import pow
from tqdm import tqdm
import glob

import numpy as np

## directory layout
## Test_set_base_path
## -- BK_Test_0
## -- -- warp_res
## -- -- occ
## -- -- GT
## -- -- ...
## -- BK_Test_1
## -- ...
TestSetBasePath = "Test_set_base_path/"
TestResultBsaePath = "Test_result_base_path/"

TestSetPath = [TestSetBasePath+"BK_Test_5/",TestSetBasePath+"BK_Test_4/",TestSetBasePath+"BK_Test_3/"] # all the test set needed for tonemapping
ResultsPath = [TestResultBsaePath+"BK_Test_5/",TestResultBsaePath+"BK_Test_4/",TestResultBsaePath+"BK_Test_3/"] # all the test set needed for tonemapping

def GetIndexRange(path, prefix):
    start = 99999
    end = 0
    for filePath in glob.glob(path + "*"):
        if prefix in filePath:
            idx = int(filePath.split('.')[1])
            start = min(start, idx)
            end = max(end, idx)
    return start, end+1



## tonemap GT
for path in TestSetPath:
    gtTonemappedPath = "Path_to_store_gt_tonemapped_results/"+path.split("/")[-2]+"/"
    if not os.path.exists(gtTonemappedPath):
            os.makedirs(gtTonemappedPath)
    imgs = os.listdir(path)
    for img in imgs:
        if "PreTonemapHDRColor" in img:
            idx = img.split(".")[1]
            os.system("mtsutil tonemap -o {}gt.{}.png {}".format(gtTonemappedPath, idx, path+img))
## tonemap results
for path in ResultsPath:
    outPath = "Path_to_store_tonemapped_results/"+path.split("/")[-2]+"/"
    if not os.path.exists(outPath):
        os.makedirs(outPath)
    

    imgs = os.listdir(path)
    for img in imgs:
        idx = img.split(".")[1]
        os.system("mtsutil tonemap -o {}res.{}.png {}".format(outPath, idx, path+img))