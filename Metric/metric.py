import glob
from os.path import dirname
import cv2
import os
import numpy as np
import math
from multiprocessing import Process
import lpips
import pytorch_fid.fid_score as pyfid
from pytorch_fid.inception import InceptionV3
import torch
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


## directory layout
## Path_to_tonemapped_GT
## -- BK_Test_0
## -- -- 1.png
## -- -- 2.png
## -- -- *.png
## -- -- ...
## -- BK_Test_1
## -- ...

## Path_to_tonemapped_ExtraNet_results
## -- BK_Test_0
## -- -- 1.png
## -- -- 2.png
## -- -- *.png
## -- -- ...
## -- BK_Test_1
## -- ...

GTPath = "Path_to_tonemapped_GT/"

comparePath = ["Path_to_tonemapped_ExtraNet_results/"]

## folders of different sets
bkList = ["BK_Test_0/","BK_Test_1/","BK_Test_2/","BK_Test_3/","BK_Test_4/"]
mdList = ["Test_1/", "Test_0/","Test_5/","Test_4/"]
frList = ["RF_Test_0/","RF_Test_1/","RF_Test_2/","RF_Test_3/","RF_Test_4/"]
wtList = ["WT_Test_0/","WT_Test_1/","WT_Test_2/","WT_Test_3/","WT_Test_4/"]

allList = [bkList,mdList, frList, wtList]#, 
listName = ["Bunker", "Medieval", "RF", "WesternTown"]#



def GetStartEnd(path):
    start = 99999
    end = 0
    for filePath in glob.glob(path + "*"):
        
        idx = int(filePath.split('.')[1])
        start = min(start, idx)
        end = max(end, idx)
    
    if start % 2 == 1:
        start += 2
    else:
        start += 1
    if end % 2 == 1:
        end -= 1
    return start, end


def doOnFolder(path):
    # for path in comparePath:
        dirName = path.split('/')[-2]
        with open("result-psnrssim.{}.txt".format(dirName), "w") as finalFile:

        
            finalFile.write(dirName + '\n')

            for i, singleList in enumerate(allList):
                totalSSIM = 0.
                totalPSNR = 0.
                totalCount = 0.0
                for singlePath in singleList:
                    
                    imgs = os.listdir(path + singlePath)
                    for imgPath in imgs:
                        idx = imgPath.split(".")[1]
                        if int(idx) % 2 != 1:                      
                            continue

                        compareImg = cv2.imread(path + singlePath + imgPath)
                        
                        gtImg = cv2.imread(GTPath + singlePath + "gt.{}.png".format(idx))
                        
                        totalSSIM += ssim(gtImg, compareImg, multichannel=True)
                        totalPSNR += calculate_psnr(gtImg, compareImg)
                        totalCount += 1.0
                
                print(listName[i] + " totalSSIM: {}, totalPSNR: {}".format(totalSSIM / totalCount, totalPSNR / totalCount))
                finalFile.write(listName[i] + " totalSSIM: {}, totalPSNR: {}\n".format(totalSSIM / totalCount, totalPSNR / totalCount))
if __name__ == "__main__":
    process_list = list()

    for path in comparePath:
       process_list.append(Process(target=doOnFolder, args=(path,)))

    for p in process_list:
       p.start()
    for p in process_list:
       p.join()
