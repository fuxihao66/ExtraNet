import cv2 as cv
cv.setNumThreads(0) 
import numpy as np
import os
def ImgRead(mPath,idx,prefix= None,format=".exr",cvtGray=False,cvtrgb=False):
    files=os.listdir(mPath)
    if prefix == None:
        prefix=files[0].split(".")[0]
    if format==".exr":
        img = cv.imread(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+format),cv.IMREAD_UNCHANGED)
    else:
        img = cv.imread(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+format))
    if cvtrgb == True:
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    if cvtGray==True:
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return img
def ImgWrite(mPath,prefix,idx,img):
    cv.imwrite(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+".exr"),img)

def ImgReadWithPrefix(mPath,idx,p=None,prefix= None,format=".exr",cvtGray=False,cvtrgb=False):
    if p == None:
        return ImgRead(mPath, idx, prefix, format, cvtGray, cvtrgb)
    files=os.listdir(mPath)
    if prefix == None:
        prefix=files[0].split(".")[0]
    if format==".exr":
        img = cv.imread(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+"."+p+format),cv.IMREAD_UNCHANGED)
    else:
        img = cv.imread(os.path.join(mPath,prefix+"."+str(idx).zfill(4)+format))
    if cvtrgb == True:
        img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    if cvtGray==True:
        img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    return img

def ReadData(path,augment=True):

    total = np.load(path)["i"]

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

    img = cv.cvtColor(img.astype(np.float32), cv.COLOR_RGB2BGR)
    img3 = cv.cvtColor(img3.astype(np.float32), cv.COLOR_RGB2BGR)
    img5 = cv.cvtColor(img5.astype(np.float32), cv.COLOR_RGB2BGR)
    imgOcc = cv.cvtColor(imgOcc.astype(np.float32), cv.COLOR_RGB2BGR)
    img_no_hole = cv.cvtColor(img_no_hole.astype(np.float32), cv.COLOR_RGB2BGR)
    img_no_hole3 = cv.cvtColor(img_no_hole3.astype(np.float32), cv.COLOR_RGB2BGR)
    img_no_hole5 = cv.cvtColor(img_no_hole5.astype(np.float32), cv.COLOR_RGB2BGR)
    gt = cv.cvtColor(gt.astype(np.float32), cv.COLOR_RGB2BGR)
    normal = cv.cvtColor(normal.astype(np.float32), cv.COLOR_RGB2BGR)

    depth = depth.astype(np.float32)
    roughness = roughness.astype(np.float32)
    metalic = metalic.astype(np.float32)

    
    depth[depth > 100] = 0.0
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    
    if augment:
        rd=np.random.uniform()
        if rd<0.2:
            return np.flip(img,0), np.flip(img3,0), np.flip(img5,0), np.flip(imgOcc,0), np.flip(img_no_hole,0), \
                   np.flip(img_no_hole3,0), np.flip(img_no_hole5,0), \
                   np.flip(gt,0), np.flip(metalic,0), np.flip(roughness,0), np.flip(depth,0), np.flip(normal,0)
        elif rd<0.3:
            return np.flip(img,1), np.flip(img3,1), np.flip(img5,1), np.flip(imgOcc,1), np.flip(img_no_hole,1), \
                   np.flip(img_no_hole3,1), np.flip(img_no_hole5,1), \
                   np.flip(gt,1), np.flip(metalic,1), np.flip(roughness,1), np.flip(depth,1), np.flip(normal,1)
        elif rd<0.35:
            return np.flip(img,(0,1)), np.flip(img3,(0,1)), np.flip(img5,(0,1)), np.flip(imgOcc,(0,1)), np.flip(img_no_hole,(0,1)), \
                   np.flip(img_no_hole3,(0,1)), np.flip(img_no_hole5,(0,1)), \
                   np.flip(gt,(0,1)), np.flip(metalic,(0,1)), np.flip(roughness,(0,1)), np.flip(depth,(0,1)), np.flip(normal,(0,1))
        else:
            return img, img3, img5, imgOcc, img_no_hole, img_no_hole3, img_no_hole5, gt, metalic, roughness, depth, normal
    else:
        return img, img3, img5, imgOcc, img_no_hole, img_no_hole3, img_no_hole5, gt, metalic, roughness, depth, normal


def ToneSimple(img):
    errors = img == -1.0
    result =  np.log(np.ones(img.shape, np.float32) + img)
    result[errors] = 0.0
    return result

def DeToneSimple(img):
    result = np.exp(img) - np.ones(img.shape, np.float32)
    result[result < 0.] = 0.
    return result
