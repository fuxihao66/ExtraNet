import shutil
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


GTPath = "Path_to_tonemapped_GT/"


comparePath = ["Path_to_tonemapped_ExtraNet_results/"]
bkList = ["BK_Test_0/","BK_Test_1/","BK_Test_2/","BK_Test_3/","BK_Test_4/"]
mdList = ["Test_1/", "Test_0/","Test_5/","Test_4/"]
frList = ["RF_Test_0/","RF_Test_1/","RF_Test_2/","RF_Test_3/","RF_Test_4/"]
wtList = ["WT_Test_0/","WT_Test_1/","WT_Test_2/","WT_Test_3/","WT_Test_4/"]
allList = [bkList,mdList, frList, wtList]#, 
listName = ["Bunker", "Medieval", "RF", "WesternTown"]#

baseImageSeqPath = "Path_to_store_temp_images/"
baseVideoPath = "Path_to_store_final_videos/"


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

def GetMinMaxIndex(imgList):
    minValue = 10000
    maxValue = -1
    for singleImg in imgList:
        idx = int(singleImg.split(".")[1])
        if idx % 2 == 0:
            continue
        minValue = min(minValue, idx)
        maxValue = max(maxValue, idx)
    return minValue, maxValue



def doOnFolder(path):
    methodName = path.split('/')[-2]
    
    videoResultPath = baseVideoPath+methodName
    videoGTPath = baseVideoPath+"GT"

    if not os.path.exists(videoResultPath):
        os.mkdir(videoResultPath)
    if not os.path.exists(videoGTPath):
        os.mkdir(videoGTPath)
    for i, singleList in enumerate(allList):
        
        for singlePath in singleList:
            tempImgSeqPath = baseImageSeqPath+methodName+'/'+singlePath.split("/")[0]
            if not os.path.exists(tempImgSeqPath):
                os.mkdir(tempImgSeqPath)


            imgs = os.listdir(path + singlePath)
            gtImgs = os.listdir(GTPath + singlePath)
            # print(imgs)

            minIndex, maxIndex = GetMinMaxIndex(gtImgs)

            minIndex = minIndex + 4
            maxIndex = maxIndex - 4

            for j in range(minIndex, maxIndex+1):
                if j % 2 == 0:
                    shutil.copyfile("{}gt.{}.png".format(GTPath + singlePath, str(j).zfill(4)),"{}/res.{}.png".format(tempImgSeqPath, str(j).zfill(4)))

                else:
                    shutil.copyfile("{}res.{}.png".format(path + singlePath, str(j).zfill(4)),"{}/res.{}.png".format(tempImgSeqPath, str(j).zfill(4)))

                    

            cmd60Result = '''ffmpeg -framerate 60 -start_number {} -i {}/res.%04d.png -c:v libx264 -preset veryslow -crf 16 {}/{}.mp4'''.format(minIndex, tempImgSeqPath, videoResultPath, methodName+'.'+singlePath.split("/")[0])
            cmd60GT = '''ffmpeg -framerate 60 -start_number {} -i {}/gt.%04d.png  -frames:v {} -c:v libx264 -preset veryslow -crf 16 {}/{}.mp4'''.format(minIndex, GTPath + singlePath, maxIndex - minIndex + 1,videoGTPath, methodName+'.'+singlePath.split("/")[0])

            os.system(cmd60Result)
            os.system(cmd60GT)
                    
if __name__ == "__main__":
    process_list = list()

    for path in comparePath:
       process_list.append(Process(target=doOnFolder, args=(path,)))

    for p in process_list:
       p.start()
    for p in process_list:
       p.join()
