import numpy as np
import cv2
import os
import sys
import glob
from multiprocessing import Process



threadNum = 4
compressedOutputDir = "I:/medieval_compressed/"
dirList = ["I:/MD_Train_6","I:/MD_Train_4","I:/MD_Train_5"]
ScenePrefix = "MedievalDocks"



WarpPrefix = "Warp"
GtPrefix = "GT"
NormalPrefix = "WorldNormal"
DepthPrefix = "SceneDepth"
MetalicPrefix = "Metallic"
RoughPrefix = "Roughness"
def MergeRange(start, end, inPath, outPath):
    for idx in range(start, end):
        newIdx = str(idx).zfill(4)
        img = cv2.imread(inPath+"/warp_res/"+ScenePrefix+WarpPrefix+".{}.1.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
        img3 = cv2.imread(inPath+"/warp_res/"+ScenePrefix+WarpPrefix+".{}.3.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
        img5 = cv2.imread(inPath+"/warp_res/"+ScenePrefix+WarpPrefix+".{}.5.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
        imgOcc = cv2.imread(inPath+"/occ/"+ScenePrefix+WarpPrefix+".{}.1.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
        img_no_hole = cv2.imread(inPath+"/warp_no_hole/"+ScenePrefix+WarpPrefix+".{}.1.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
        img_no_hole3 = cv2.imread(inPath+"/warp_no_hole/"+ScenePrefix+WarpPrefix+".{}.3.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
        img_no_hole5 = cv2.imread(inPath+"/warp_no_hole/"+ScenePrefix+WarpPrefix+".{}.5.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
        
        gt = cv2.imread(inPath+"/GT/"+ScenePrefix+GtPrefix+".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)
        metalic = cv2.imread(inPath+"/"+ScenePrefix+MetalicPrefix+".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)[:,:,0:1]
        roughness = cv2.imread(inPath+"/"+ScenePrefix+RoughPrefix+".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)[:,:,0:1]
        depth = cv2.imread(inPath+"/"+ScenePrefix+DepthPrefix+".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)[:,:,0:1]
        normal = cv2.imread(inPath+"/"+ScenePrefix+NormalPrefix+".{}.exr".format(newIdx), cv2.IMREAD_UNCHANGED)


        res = np.concatenate([img,img3,img5, imgOcc, img_no_hole, img_no_hole3, img_no_hole5, gt, metalic, roughness, depth, normal], axis=2)
        res = res.astype(np.float16)

        np.savez_compressed(outPath+'compressed.{}.npz'.format(newIdx), i = res)
        # np.save(outPath+'compressed.{}'.format(newIdx), res)
def GetCompressStartEnd(path):
    start = 99999
    end = 0

    for filePath in glob.glob(path + "/*"):
        idx = int(filePath.split('.')[1])
        start = min(start, idx)
        end = max(end, idx)
    return start, end
def GetStartEnd(path):
    start = 99999
    end = 0

    for filePath in glob.glob(path + "/GT/*"):
        idx = int(filePath.split('.')[1])
        start = min(start, idx)
        end = max(end, idx)
    return start, end
def MergeFile(inPath, outPath):
    # get index range
    start, end = GetStartEnd(inPath)

    if not os.path.exists(outPath):
        os.mkdir(outPath)
    # combine
    processList = list()
    part = (end - start + 1) // threadNum + 1
    for t in range(threadNum):
        processList.append(Process(target=MergeRange, args=(start+t*part, min(start+t*part+part, end+1),inPath, outPath,)))

    for sub_process in processList:
        sub_process.start()
    for sub_process in processList:
        sub_process.join()

def ReadData(path):

    total = np.load(path)

    img = total[:,:,0:3]
    img3 = total[:,:,3:6]
    img5 = total[:,:,6:9]
    imgOcc = total[:,:,9:12]
    img_no_hole = total[:,:,12:15]
    img_no_hole3 = total[:,:,15:18]
    img_no_hole5 = total[:,:,18:21]
    gt = total[:,:,21:24]
    metalic = total[:,:,24:25]
    roughness = total[:,:,25:26]
    depth = total[:,:,26:27]
    normal = total[:,:,27:30]

    img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2BGR)
    img3 = cv2.cvtColor(img3.astype(np.float32), cv2.COLOR_RGB2BGR)
    img5 = cv2.cvtColor(img5.astype(np.float32), cv2.COLOR_RGB2BGR)
    imgOcc = cv2.cvtColor(imgOcc.astype(np.float32), cv2.COLOR_RGB2BGR)
    img_no_hole = cv2.cvtColor(img_no_hole.astype(np.float32), cv2.COLOR_RGB2BGR)
    img_no_hole3 = cv2.cvtColor(img_no_hole3.astype(np.float32), cv2.COLOR_RGB2BGR)
    img_no_hole5 = cv2.cvtColor(img_no_hole5.astype(np.float32), cv2.COLOR_RGB2BGR)
    gt = cv2.cvtColor(gt.astype(np.float32), cv2.COLOR_RGB2BGR)
    normal = cv2.cvtColor(normal.astype(np.float32), cv2.COLOR_RGB2BGR)
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    

    # cv2.imwrite("normal.exr", normal.astype(np.float32))
    # cv2.imwrite("img.exr", img)
    # cv2.imwrite("img3.exr", img3)
    # cv2.imwrite("img5.exr", img5)
    # cv2.imwrite("imgOcc.exr", imgOcc)
    # cv2.imwrite("img_no_hole.exr", img_no_hole)
    # cv2.imwrite("img_no_hole3.exr", img_no_hole3)
    # cv2.imwrite("img_no_hole5.exr", img_no_hole5)
    # cv2.imwrite("gt.exr", gt)


    return img, img3, img5, imgOcc, img_no_hole, img_no_hole3, img_no_hole5, gt, metalic, roughness, depth, normal

def Compress32To16(start, end, path):
    for idx in range(start, end):
        fileName = path+"/compressed.{}.npy".format(str(idx).zfill(4))
        try:
            total = np.load(fileName)
        except:
            continue
        total = total.astype(np.float16)
        np.save(fileName, total)
def CompressRange(di):
    start, end = GetCompressStartEnd(di)
    print(start)
    print(end)
    # combine
    processList = list()
    part = (end - start + 1) // threadNum + 1
    for t in range(threadNum):
        processList.append(Process(target=Compress32To16, args=(start+t*part, min(start+t*part+part, end+1),di,)))

    for sub_process in processList:
        sub_process.start()
    for sub_process in processList:
        sub_process.join()

if __name__ == "__main__":
    for di in dirList:
        MergeFile(di, compressedOutputDir+di.split("/")[-1]+"/")
    # for di in dirList:
    #     CompressRange(di)

    # Compress32To16(r"D:\NoAACompressed\Medieval_0\compressed.0011.npy")
    # ReadData(r"D:\NoAACompressed\Medieval_0\compressed.0011.npy")
    # ReadData(r"E:\taa\compressed.0180.npy")