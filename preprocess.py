import numpy as np
import cv2 as cv
import os
import sys
import glob
from multiprocessing import Process
# from win32com.shell import shellcon, shell

ENABLE_OCCWARP = True
ENABLE_DEMODULATE = True

Stencil_Only = False
Normal_Only = False
WorldPos_Only = False
Warp_Only = True
DEBUG_OCC = False
DEBUG_MV = True

WARP_NUM = 5
Thread_NUM = 8

class GlobalInfo():

    def __init__(self, path, name):

        # setup path
        self.mPath = path
        self.mGTPath = os.path.join(path, "GT")
        self.mResPath = os.path.join(path, "warp_res")
        self.mDemodulatePath = os.path.join(path, "demodulate")
        self.mOCCPath = os.path.join(path, "occ")
        self.mOCCWarpPath = os.path.join(path, "OCCDebug")
        self.mWarpMotionPath = os.path.join(path, "WarpMotion")
        self.mResWPPath = os.path.join(path, "WorldPosDebug")
        self.mResSPath = os.path.join(path, "StencilDebug")
        self.mResNPath = os.path.join(path, "NormalDebug")
        self.mNoHolePath = os.path.join(path, "warp_no_hole")

        if os.path.exists(self.mResPath) is False:
            os.makedirs(self.mResPath)

        if os.path.exists(self.mGTPath) is False:
            os.makedirs(self.mGTPath)

        if ENABLE_DEMODULATE:
            if os.path.exists(self.mDemodulatePath) is False:
                os.makedirs(self.mDemodulatePath)

        if ENABLE_OCCWARP:
            if os.path.exists(self.mOCCPath) is False:
                os.makedirs(self.mOCCPath)

            if DEBUG_OCC:
                if os.path.exists(self.mOCCWarpPath) is False:
                    os.makedirs(self.mOCCWarpPath)

        if DEBUG_MV:
            if os.path.exists(self.mWarpMotionPath) is False:
                os.makedirs(self.mWarpMotionPath)

        if WorldPos_Only:
            if os.path.exists(self.mResWPPath) is False:
                os.makedirs(self.mResWPPath)

        if Stencil_Only:
            if os.path.exists(self.mResSPath) is False:
                os.makedirs(self.mResSPath)

        if Normal_Only:
            if os.path.exists(self.mResNPath) is False:
                os.makedirs(self.mResNPath)

        if Warp_Only:
            if os.path.exists(self.mNoHolePath) is False:
                os.makedirs(self.mNoHolePath)

        # setup prefix
        self.PrefixSN = name
        self.PrefixFI = name + "FinalImage."
        self.PrefixWN = name + "WorldNormal."
        self.PrefixWP = name + "WorldPosition."
        self.PrefixSD = name + "SceneDepth."
        self.PrefixMV = name + "MotionVector."
        self.PrefixSC = name + "MyStencil."
        self.PrefixNV = name + "NoV."
        self.PrefixSpe = name + "Specular."
        self.PrefixMetallic = name + "Metallic."
        # self.PrefixBC = name + "BaseColorAA."
        self.PrefixBC = name + "BaseColor."
        self.PrefixPreTM = name + "PreTonemapHDRColor."
        self.PrefixWarp = name + "Warp."
        self.PrefixGT = name + "GT."

        # setup merge files
        self.PrefixR = name + "Roughness."
        self.PrefixM = name + "Metallic."
        self.mMergePath = os.path.join(path, "merged_files")
        self.mFinal = os.path.join(self.mMergePath, "final")
        self.mGbuffer = os.path.join(self.mMergePath, "gbuffer")
        self.mMergeWarpRes = os.path.join(self.mFinal, "warp_res")
        self.mMergeGT = os.path.join(self.mFinal, "GT")
        self.mMergeOCC = os.path.join(self.mFinal, "occ")
        self.mMergeNoHole = os.path.join(self.mFinal, "warp_no_hole")

        if os.path.exists(self.mMergePath) is False:
            os.makedirs(self.mMergePath)
        if os.path.exists(self.mFinal) is False:
            os.makedirs(self.mFinal)
        if os.path.exists(self.mGbuffer) is False:
            os.makedirs(self.mGbuffer)        
        if os.path.exists(self.mMergeWarpRes) is False:
            os.makedirs(self.mMergeWarpRes)
        if os.path.exists(self.mMergeGT) is False:
            os.makedirs(self.mMergeGT)
        if os.path.exists(self.mMergeOCC) is False:
            os.makedirs(self.mMergeOCC)
        if os.path.exists(self.mMergeNoHole) is False:
            os.makedirs(self.mMergeNoHole)


def init(path):

    PrefixSN = ""
    PrefixPreTM = "PreTonemapHDRColor."
    for filePath in glob.glob(path + "\\*"):
        if PrefixPreTM in filePath:
            PrefixSN = filePath.split('\\')[-1].split(PrefixPreTM)[0]
            break

    globalInfo = GlobalInfo(path, PrefixSN)

    start = 9999
    end = 0

    for filePath in glob.glob(path + "/*"):
        if PrefixPreTM in filePath:
            idx = int(filePath.split('.')[1])
            start = min(start, idx)
            end = max(end, idx)

    print("Scene name: ", PrefixSN, "start: ", start, "end: ", end)

    return globalInfo, start, end

def check(globalInfo, start, end):
    print("start checking files from", start, " to ", end + 1)
    check_prefix_list = list()
    check_prefix_list.append(globalInfo.PrefixPreTM)
    check_prefix_list.append(globalInfo.PrefixBC)
    check_prefix_list.append(globalInfo.PrefixWN)
    check_prefix_list.append(globalInfo.PrefixWP)
    check_prefix_list.append(globalInfo.PrefixSD)
    check_prefix_list.append(globalInfo.PrefixMV)
    check_prefix_list.append(globalInfo.PrefixSC)
    check_prefix_list.append(globalInfo.PrefixNV)

    check_res = True
    for i in range(start, end + 1):
        idx = str(i).zfill(4)
        for check_prefix in check_prefix_list:
            filepath = os.path.join(globalInfo.mPath, check_prefix + idx + ".exr")
            if os.path.exists(filepath) is False:
                print("check file failed: ", filepath)
                check_res = False

    if check_res:
        print("check files passed")
    return check_res

def demodulate(id, globalInfo, start, end):

    print("Process-", id, " demodulate from", start, " to ", end)
    
    fix_start = start
    for filePath in glob.glob(globalInfo.mDemodulatePath + "/*"):
        if globalInfo.PrefixPreTM in filePath:
            idx = int(filePath.split('.')[1])
            if idx < end and idx >= fix_start:
                fix_start = idx + 1

    print("Process-", id, ": fix start from ", start, " to ", fix_start)

    for i in range(fix_start, end):
        idx = str(i).zfill(4)
        # idx_next = str(i + 1).zfill(4)
        img = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixPreTM + idx + ".exr"), cv.IMREAD_UNCHANGED)
        # img_BaseColor = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixBC + idx_next + ".exr"), cv.IMREAD_UNCHANGED)
        img_BaseColor = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixBC + idx + ".exr"), cv.IMREAD_UNCHANGED)

        img_Res = img / img_BaseColor
        img_Res[img_BaseColor == 0] = 0
        cv.imwrite(os.path.join(globalInfo.mDemodulatePath, globalInfo.PrefixPreTM + idx + ".exr"), img_Res)
        
        print("Process-", id, ": finish ", idx)
    print("Process-", id, ": end process ",)


def lerp(a, b, alpha):
    return a * alpha + b * (1 - alpha)


def Linear_Warp(img, height, width, warp2_i0, warp2_j0, warp2_i1, warp2_j1, weight2_i0, weight2_i1, weight2_j0,
                weight2_j1):
    res2_i0j0 = img[warp2_i0, warp2_j0]
    res2_i1j0 = img[warp2_i1, warp2_j0]
    res2_i0j1 = img[warp2_i0, warp2_j1]
    res2_i1j1 = img[warp2_i1, warp2_j1]

    res2_j0 = res2_i0j0 * weight2_i0 + res2_i1j0 * weight2_i1
    res2_j1 = res2_i0j1 * weight2_i0 + res2_i1j1 * weight2_i1
    res2 = res2_j0 * weight2_j0 + res2_j1 * weight2_j1
    res2 = res2.reshape((height, width, 3)).astype(np.float32)

    return res2


def Nearly_Warp(img, height, width, warp_i, warp_j):
    res = img[warp_i, warp_j]
    res = res.reshape((height, width, 3)).astype(np.float32)
    return res


def merge_by_max(matrix0, matrix1):
    res = np.zeros([matrix0.shape[0], matrix0.shape[1]]).astype(np.float32)
    res[matrix0 > matrix1] = matrix0[matrix0 > matrix1]
    res[matrix1 >= matrix0] = matrix1[matrix1 >= matrix0]
    return res


def merge_by_longest(mv0, mv1):
    length0 = np.sqrt(np.square(mv0).sum(axis=-1, keepdims=True)).repeat(3, axis=-1)
    length1 = np.sqrt(np.square(mv1).sum(axis=-1, keepdims=True)).repeat(3, axis=-1)
    mv0[length1 > length0] = mv1[length1 > length0]
    zero = np.logical_or(length0 == 0, length1 == 0)
    mv0[zero] = 0
    return mv0


def warp_motion_vector(motion_vector_org, motion_vector_use):
    height, width, _ = motion_vector_org.shape
    # here we use index to implement warp
    flat_index = np.arange(height * width)
    i = flat_index // width
    j = flat_index - i * width
    # plus one for the padding
    i += 1
    j += 1

    motion_vector_org_pad = np.pad(motion_vector_org, ((1, 1), (1, 1), (0, 0)), constant_values=0.0)
    motion_vector_use = np.pad(motion_vector_use, ((1, 1), (1, 1), (0, 0)), constant_values=0.0)

    warp2_i = np.clip(i + motion_vector_org_pad[i, j, 1], 0, height + 1).astype(np.long)
    warp2_j = np.clip(j - motion_vector_org_pad[i, j, 0], 0, width + 1).astype(np.long)

    # linear interpolation, note that the idx that is out of the boundry will be clamped and then pad to zero
    warp2_i0 = np.clip(np.floor(warp2_i), 0, height + 1).astype(np.long)
    warp2_i1 = np.clip(np.floor(warp2_i) + 1, 0, height + 1).astype(np.long)
    warp2_j0 = np.clip(np.floor(warp2_j), 0, width + 1).astype(np.long)
    warp2_j1 = np.clip(np.floor(warp2_j) + 1, 0, width + 1).astype(np.long)

    warp_i0j0 = motion_vector_use[warp2_i0, warp2_j0]
    warp_i0j1 = motion_vector_use[warp2_i0, warp2_j1]
    warp_i1j0 = motion_vector_use[warp2_i1, warp2_j0]
    warp_i1j1 = motion_vector_use[warp2_i1, warp2_j1]

    warp_i0j0 = warp_i0j0.reshape((height, width, 3)).astype(np.float32)
    warp_i0j1 = warp_i0j1.reshape((height, width, 3)).astype(np.float32)
    warp_i1j0 = warp_i1j0.reshape((height, width, 3)).astype(np.float32)
    warp_i1j1 = warp_i1j1.reshape((height, width, 3)).astype(np.float32)

    res = merge_by_longest(warp_i0j0, warp_i0j1)
    res = merge_by_longest(res, warp_i1j0)
    res = merge_by_longest(res, warp_i1j1)

    return np.add(motion_vector_org, res)


def min_world_pos_distance(world_pos0, world_pos1, height, width, warp2_i0, warp2_j0, warp2_i1, warp2_j1):
    warp_i0j0 = world_pos0[warp2_i0, warp2_j0]
    warp_i0j1 = world_pos0[warp2_i0, warp2_j1]
    warp_i1j0 = world_pos0[warp2_i1, warp2_j0]
    warp_i1j1 = world_pos0[warp2_i1, warp2_j1]

    warp_i0j0 = warp_i0j0.reshape((height, width, 3)).astype(np.float32)
    warp_i0j1 = warp_i0j1.reshape((height, width, 3)).astype(np.float32)
    warp_i1j0 = warp_i1j0.reshape((height, width, 3)).astype(np.float32)
    warp_i1j1 = warp_i1j1.reshape((height, width, 3)).astype(np.float32)

    distance00 = np.sqrt(np.square(warp_i0j0 - world_pos1).sum(axis=-1, keepdims=True)).reshape((height, width)).astype(
        np.float32)
    distance01 = np.sqrt(np.square(warp_i0j1 - world_pos1).sum(axis=-1, keepdims=True)).reshape((height, width)).astype(
        np.float32)
    distance10 = np.sqrt(np.square(warp_i1j0 - world_pos1).sum(axis=-1, keepdims=True)).reshape((height, width)).astype(
        np.float32)
    distance11 = np.sqrt(np.square(warp_i1j1 - world_pos1).sum(axis=-1, keepdims=True)).reshape((height, width)).astype(
        np.float32)

    res = merge_by_max(distance00, distance01)
    res = merge_by_max(res, distance10)
    res = merge_by_max(res, distance11)
    return res


def GetOCCMV(motion2, depthCurr, depthPrev=None,albedoCurr=None, albedoPrev=None ):
    # warp the motion vector back so that we can use occ motion vector
    height, width, _ = motion2.shape

    occmv = np.zeros((height, width, _))

    cullingBuffer = np.zeros((height, width))
    cullingBuffer[cullingBuffer == 0.] = 10.
    # back warp scattering
    for i in range(height):
        for j in range(width):
            samplePos = np.array([i + motion2[i, j, 1], j - motion2[i, j, 0]])
            # check four neighbour, overwrite them if the len of this mv is bigger than the other
            Pos00 = np.floor(samplePos).astype(np.int)
            Pos10 = np.array([Pos00[0] + 1, Pos00[1]])
            Pos01 = np.array([Pos00[0], Pos00[1] + 1])
            Pos11 = np.array([Pos00[0] + 1, Pos00[1] + 1])
            if Pos00[0] < 0 or Pos00[0] >= height or Pos00[1] < 0 or Pos00[1] >= width:
                pass
            else:
                # if  colorClose(albedoCurr[i, j], albedoPrev[Pos00[0], Pos00[1]]) :
                if cullingBuffer[Pos00[0], Pos00[1]] > depthCurr[i, j][0]:
                    cullingBuffer[Pos00[0], Pos00[1]] = depthCurr[i, j][0]
                    occmv[Pos00[0], Pos00[1]] = motion2[i, j]
            if Pos01[0] < 0 or Pos01[0] >= height or Pos01[1] < 0 or Pos01[1] >= width:
                pass
            else:
                # if np.square(motion2[i, j]).sum() > np.square(occmv[Pos01[0], Pos01[1]]).sum():
                # if  colorClose(albedoCurr[i, j], albedoPrev[Pos01[0], Pos01[1]]) :
                if cullingBuffer[Pos01[0], Pos01[1]] > depthCurr[i, j][0]:
                    cullingBuffer[Pos01[0], Pos01[1]] = depthCurr[i, j][0]
                    occmv[Pos01[0], Pos01[1]] = motion2[i, j]
            if Pos10[0] < 0 or Pos10[0] >= height or Pos10[1] < 0 or Pos10[1] >= width:
                pass
            else:
                # if np.square(motion2[i, j]).sum() > np.square(occmv[Pos10[0], Pos10[1]]).sum():
                # if  colorClose(albedoCurr[i, j], albedoPrev[Pos10[0], Pos10[1]]) :
                if cullingBuffer[Pos10[0], Pos10[1]] > depthCurr[i, j][0]:
                    cullingBuffer[Pos10[0], Pos10[1]] = depthCurr[i, j][0]
                    occmv[Pos10[0], Pos10[1]] = motion2[i, j]
            if Pos11[0] < 0 or Pos11[0] >= height or Pos11[1] < 0 or Pos11[1] >= width:
                pass
            else:
                # if np.square(motion2[i, j]).sum() > np.square(occmv[Pos11[0], Pos11[1]]).sum():
                # if  colorClose(albedoCurr[i, j], albedoPrev[Pos11[0], Pos11[1]]) :
                if cullingBuffer[Pos11[0], Pos11[1]] > depthCurr[i, j][0]:
                    cullingBuffer[Pos11[0], Pos11[1]] = depthCurr[i, j][0]
                    occmv[Pos11[0], Pos11[1]] = motion2[i, j]
    # final mv generate
    finalmv = np.zeros((height, width, _))

    flat_index = np.arange(height * width)
    i = flat_index // width
    j = flat_index - i * width

    warp2_i = np.clip(np.floor(i + motion2[i, j, 1]), 0, height - 1).astype(np.long)
    warp2_j = np.clip(np.floor(j - motion2[i, j, 0]), 0, width - 1).astype(np.long)

    finalmv = occmv[warp2_i, warp2_j]
    finalmv = finalmv.reshape((height, width, 3)).astype(np.float32)
    
    
    return finalmv

def getOCCWarpImg(img, occ_motion_vector, height, width):

    flat_index = np.arange(height * width)
    i = flat_index // width
    j = flat_index - i * width

    i += 1
    j += 1

    warp2_i = i + occ_motion_vector[i, j, 1]
    warp2_j = j - occ_motion_vector[i, j, 0]


    # linear interpolation, note that the idx that is out of the boundry will be clamped and then pad to zero
    warp2_i0 = np.floor(warp2_i)
    warp2_i1 = np.floor(warp2_i) + 1
    warp2_j0 = np.floor(warp2_j)
    warp2_j1 = np.floor(warp2_j) + 1

    weight2_i0 = (1 - np.abs(warp2_i0 - warp2_i))[:, np.newaxis]
    weight2_i1 = (1 - np.abs(warp2_i1 - warp2_i))[:, np.newaxis]
    weight2_j0 = (1 - np.abs(warp2_j0 - warp2_j))[:, np.newaxis]
    weight2_j1 = (1 - np.abs(warp2_j1 - warp2_j))[:, np.newaxis]

    warp2_i0 = np.clip(warp2_i0, 0, height + 1).astype(np.long)
    warp2_i1 = np.clip(warp2_i1, 0, height + 1).astype(np.long)
    warp2_j0 = np.clip(warp2_j0, 0, width + 1).astype(np.long)
    warp2_j1 = np.clip(warp2_j1, 0, width + 1).astype(np.long)

    res = Linear_Warp(img, height, width, warp2_i0, warp2_j0, warp2_i1, warp2_j1,
                                   weight2_i0, weight2_i1, weight2_j0, weight2_j1)

    return res

def make_hole(id, globalInfo, start, end):

    print("Process-", id, ": make hole from", start, " to ", end)

    # fix_end = end
    # for filePath in glob.glob(globalInfo.mGTPath + "/*"):
    #     if globalInfo.PrefixGT in filePath:
    #         idx = int(filePath.split('.')[1])
    #         if idx > start:
    #             fix_end = min(fix_end, idx)
    # print("Process-", id, ": fix end from ", end, " to ", fix_end)
    # end = fix_end

    fix_start = start
    for filePath in glob.glob(globalInfo.mGTPath + "/*"):
        if globalInfo.PrefixGT in filePath:
            idx = int(filePath.split('.')[1])
            if idx < end and idx >= fix_start:
                fix_start = idx + 1

    print("Process-", id, ": fix start from ", start, " to ", fix_start)

    for idx in range(fix_start, end):

        img = list()
        world_normal = list()
        world_position = list()
        custom_stencil = list()
        motion_vector = list()

        for j in range(WARP_NUM):
            idx_cur = str(idx + j).zfill(4)
            idx_mv = str(idx + j + 1).zfill(4)
            if ENABLE_DEMODULATE:
                img.append(cv.imread(os.path.join(globalInfo.mDemodulatePath, globalInfo.PrefixPreTM + idx_cur + ".exr"), cv.IMREAD_UNCHANGED))
            else:
                img.append(cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixPreTM + idx_cur + ".exr"), cv.IMREAD_UNCHANGED))
            world_normal.append(cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixWN + idx_cur + ".exr"), cv.IMREAD_UNCHANGED))
            world_position.append(cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixWP + idx_cur + ".exr"), cv.IMREAD_UNCHANGED))
            custom_stencil.append(cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixSC + idx_cur + ".exr"), cv.IMREAD_UNCHANGED))
            motion_vector_temp = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixMV + idx_mv + ".exr"), cv.IMREAD_UNCHANGED)
            motion_vector.append(cv.cvtColor(motion_vector_temp, cv.COLOR_BGR2RGB))

        idx_res = str(idx + WARP_NUM).zfill(4)
        depth_target = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixSD + idx_res + ".exr"), cv.IMREAD_UNCHANGED)
        NoV_target = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixNV + idx_res + ".exr"), cv.IMREAD_UNCHANGED)
        world_normal_target = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixWN + idx_res + ".exr"), cv.IMREAD_UNCHANGED)
        world_position_target = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixWP + idx_res + ".exr"), cv.IMREAD_UNCHANGED)
        custom_stencil_target = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixSC + idx_res + ".exr"), cv.IMREAD_UNCHANGED)

        if ENABLE_DEMODULATE:
            img_gt = cv.imread(os.path.join(globalInfo.mDemodulatePath, globalInfo.PrefixPreTM + idx_res + ".exr"), cv.IMREAD_UNCHANGED)
        else:
            img_gt = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixPreTM + idx_res + ".exr"), cv.IMREAD_UNCHANGED)

        world_normal_target_len = np.sqrt(np.square(world_normal_target).sum(axis=-1, keepdims=True)).repeat(3, axis=-1)

        height, width, _ = depth_target.shape

        for t in range(WARP_NUM):
            if t % 2 != 0:
                continue
            motion_vector_temp = motion_vector[t]
            for mv_t in range(t + 1, WARP_NUM):
                motion_vector_temp = warp_motion_vector(motion_vector_temp, motion_vector[mv_t])

            # pad the images so the hole in the warpped frame is white

            img_cur = np.pad(img[t], ((1, 1), (1, 1), (0, 0)), constant_values=-10.0)
            motion_vector_cur = np.pad(motion_vector_temp, ((1, 1), (1, 1), (0, 0)), constant_values=0.0)

            world_normal_cur = np.pad(world_normal[t], ((1, 1), (1, 1), (0, 0)), constant_values=0)
            world_position_cur = np.pad(world_position[t], ((1, 1), (1, 1), (0, 0)), constant_values=1e5)
            custom_stencil_cur = np.pad(custom_stencil[t], ((1, 1), (1, 1), (0, 0)), constant_values=0.0)

            # here we use index to implement warp
            flat_index = np.arange(height * width)
            i = flat_index // width
            j = flat_index - i * width
            # plus one for the padding
            i += 1
            j += 1
            # motion vector record the change in SCREEN SPACE! and indexed by XY!
            # ↑
            # ↑
            # ↑
            # ↑
            # ↑
            # ↑
            # ↑ → → → → → → → →
            # when we index the array by i, j the space is like that
            # ↓ → → → → → → → →
            # ↓
            # ↓
            # ↓
            # ↓
            # ↓
            # ↓
            # keep in mind with that

            warp2_i = i + motion_vector_cur[i, j, 1]
            warp2_j = j - motion_vector_cur[i, j, 0]

            # warp2_i = np.clip(i + motion_vector_cur[i, j, 1], 0, height + 1).astype(np.long)
            # warp2_j = np.clip(j - motion_vector_cur[i, j, 0], 0, width + 1).astype(np.long)

            # linear interpolation, note that the idx that is out of the boundry will be clamped and then pad to zero
            warp2_i0 = np.floor(warp2_i)
            warp2_i1 = np.floor(warp2_i) + 1
            warp2_j0 = np.floor(warp2_j)
            warp2_j1 = np.floor(warp2_j) + 1

            weight2_i0 = (1 - np.abs(warp2_i0 - warp2_i))[:, np.newaxis]
            weight2_i1 = (1 - np.abs(warp2_i1 - warp2_i))[:, np.newaxis]
            weight2_j0 = (1 - np.abs(warp2_j0 - warp2_j))[:, np.newaxis]
            weight2_j1 = (1 - np.abs(warp2_j1 - warp2_j))[:, np.newaxis]

            warp2_i0 = np.clip(warp2_i0, 0, height + 1).astype(np.long)
            warp2_i1 = np.clip(warp2_i1, 0, height + 1).astype(np.long)
            warp2_j0 = np.clip(warp2_j0, 0, width + 1).astype(np.long)
            warp2_j1 = np.clip(warp2_j1, 0, width + 1).astype(np.long)


            res_warp_img = Linear_Warp(img_cur, height, width, warp2_i0, warp2_j0, warp2_i1, warp2_j1, weight2_i0, weight2_i1, weight2_j0, weight2_j1)
            normal_warp = Linear_Warp(world_normal_cur, height, width, warp2_i0, warp2_j0, warp2_i1, warp2_j1, weight2_i0, weight2_i1, weight2_j0, weight2_j1)
            # world_position_warp = Linear_Warp(img0_world_position, height, width, warp2_i0, warp2_j0, warp2_i1, warp2_j1, weight2_i0, weight2_i1, weight2_j0, weight2_j1)
            custom_stencil_warp = Linear_Warp(custom_stencil_cur, height, width, warp2_i0, warp2_j0, warp2_i1, warp2_j1, weight2_i0, weight2_i1, weight2_j0, weight2_j1)

            res_warp_img[res_warp_img < 0] = -1

            # moving actor
            normal_warp_len = np.sqrt(np.square(normal_warp).sum(axis=-1, keepdims=True)).repeat(3, axis=-1)

            normal_warp /= (normal_warp_len + 1e-5)
            cos_normal = (normal_warp * world_normal_target).sum(axis=-1, keepdims=True).repeat(3, axis=-1)
            # normal difference
            normal_difference = cos_normal < 0.98
            # false only when both are zero
            empty = np.logical_or(normal_warp_len != 0, world_normal_target_len != 0)
            normal_difference = np.logical_and(normal_difference, empty)
            b_moving_actor = custom_stencil_warp != 0
            moving_actor_hole = np.logical_and(normal_difference, b_moving_actor)

            # static actor
            bias = lerp(7.5, 45, np.abs(NoV_target[:, :, 0])) + depth_target[:, :, 0] * 50
            # world_position_distance = np.sqrt(np.square(world_position_warp - img1_world_position).sum(axis=-1, keepdims=True)).repeat(3, axis=-1)
            world_position_distance = min_world_pos_distance(world_position_cur, world_position_target, height, width, warp2_i0, warp2_j0, warp2_i1, warp2_j1)
            world_position_diff = (world_position_distance > bias).repeat(3, axis=-1).reshape(height, width, 3)
            b_static_actor = custom_stencil_warp == 0
            static_disocc = np.logical_and(world_position_diff, b_static_actor)

            # hole for moving actor
            mesh_minus = np.abs(custom_stencil_warp - custom_stencil_target).sum(axis=-1, keepdims=True).repeat(3, axis=-1)
            mesh_hole = mesh_minus > 1

            # hole = np.logical_or(normal_difference, disocc, mesh_diff)
            hole = np.logical_or(static_disocc, mesh_hole)
            hole = np.logical_or(hole, moving_actor_hole)

            # for Debug
            if Normal_Only:
                moving_actor_only_warp = res_warp_img.copy()
                moving_actor_only_warp[moving_actor_hole] = -1
                cv.imwrite(os.path.join(globalInfo.mResNPath, "NormalOnlyWarp" + idx_res + '.' + str(WARP_NUM - t) + ".exr"), moving_actor_only_warp)

            if WorldPos_Only:
                world_pos_only_warp = res_warp_img.copy()
                world_pos_only_warp[static_disocc] = -1
                cv.imwrite(os.path.join(globalInfo.mResWPPath, "WorldPosOnlyWarp." + idx_res + '.' + str(WARP_NUM - t) + ".exr"), world_pos_only_warp)
                cv.imwrite(os.path.join(globalInfo.mResWPPath, "worldPosDiff." + idx_res + '.' + str(WARP_NUM - t) + ".exr"), world_position_distance)
                cv.imwrite(os.path.join(globalInfo.mResWPPath, "worldPosBias." + idx_res + '.' + str(WARP_NUM - t) + ".exr"), bias)

            if Stencil_Only:
                stencil_only_warp = res_warp_img.copy()
                stencil_only_warp[mesh_hole] = -1
                cv.imwrite(os.path.join(globalInfo.mResSPath, "StencilOnlywarpDiffuse." + idx_res + '.' + str(WARP_NUM - t) + ".exr"), stencil_only_warp)

            if Warp_Only:
                cv.imwrite(os.path.join(globalInfo.mNoHolePath, globalInfo.PrefixWarp + idx_res + '.' + str(WARP_NUM - t) + ".exr"), res_warp_img)

            res_warp_img[hole] = -1
            cv.imwrite(os.path.join(globalInfo.mResPath, globalInfo.PrefixWarp + idx_res + '.' + str(WARP_NUM - t) + ".exr"), res_warp_img)

            if t == WARP_NUM - 1:
                if ENABLE_OCCWARP:
                    prev_depth = cv.imread(os.path.join(globalInfo.mPath, globalInfo.PrefixSD + str(idx + t) + ".exr"), cv.IMREAD_UNCHANGED)
                    occ_motion_vector = GetOCCMV(motion_vector_temp, depth_target)
                    if DEBUG_OCC:
                        cv.imwrite(os.path.join(globalInfo.mOCCWarpPath, "OCCMotionVector" + idx_res + '.' + str(WARP_NUM - t) + ".exr"), occ_motion_vector)

                    occ_motion_vector = np.pad(occ_motion_vector, ((1, 1), (1, 1), (0, 0)), constant_values=0.0)

                    occ_img_warp = getOCCWarpImg(img_cur, occ_motion_vector, height, width)
                    res_warp_img[hole] = occ_img_warp[hole]
                    cv.imwrite(os.path.join(globalInfo.mOCCPath, globalInfo.PrefixWarp + idx_res + '.' + str(WARP_NUM - t) + ".exr"), res_warp_img)
                    if DEBUG_OCC:
                       cv.imwrite(os.path.join(globalInfo.mOCCWarpPath, "OCCWarpImg" + idx_res + '.' + str(WARP_NUM - t) + ".exr"), occ_img_warp)

                if DEBUG_MV:
                    motion_vector_temp = cv.cvtColor(motion_vector_temp, cv.COLOR_RGB2BGR)
                    cv.imwrite(os.path.join(globalInfo.mWarpMotionPath, globalInfo.PrefixMV + idx_res + '.' + str(WARP_NUM - t) + ".exr"), motion_vector_temp)

            print("Process-", id, ": output:", str(idx + t), " to ", idx_res)

        cv.imwrite(os.path.join(globalInfo.mGTPath, globalInfo.PrefixGT + idx_res + ".exr"), img_gt)

        print("Process-", id, ": finish ", idx_res)

    print("Process-", id, ": finish warp ", start, " to ", end)


def merge_files(globalInfo, start, end):

    print("merge gbuffer from", start, " to ", end)

    fix_start = start
    for filePath in glob.glob(globalInfo.mGbuffer + "/*"):
        if globalInfo.PrefixBC in filePath:
            idx = int(filePath.split('.')[1])
            if idx < end and idx >= fix_start:
                fix_start = idx + 1

    print("fix start from ", start, " to ", fix_start)

    # Gbuffer
    Gbuffer_list = list()
    Gbuffer_list.append(globalInfo.PrefixWN)
    Gbuffer_list.append(globalInfo.PrefixSD)
    Gbuffer_list.append(globalInfo.PrefixR)
    Gbuffer_list.append(globalInfo.PrefixM)

    for i in range(fix_start, end):
        idx = str(i).zfill(4)
        # idx_next = str(i + 1).zfill(4)
        for gbuffer_prefix in Gbuffer_list:
            src_filepath = globalInfo.mPath + "\\" + gbuffer_prefix + idx + ".exr"
            dst_filepath = globalInfo.mGbuffer + "\\" + gbuffer_prefix + idx + ".exr"
            
            # print (src_filepath, " => ", dst_filepath)
            os.system ("copy %s %s" % (src_filepath, dst_filepath))
            # shell.SHFileOperation((0, shellcon.FO_COPY, src_filepath, dst_filepath, shellcon.FOF_NOCONFIRMMKDIR, None, None))
            # copyfile(src_filepath, dst_filepath)
        
        # BaseColor is prev frame
        src_filepath = globalInfo.mPath + "\\" + globalInfo.PrefixBC + idx + ".exr"
        dst_filepath = globalInfo.mGbuffer + "\\" + globalInfo.PrefixBC + idx + ".exr"
        os.system ("copy %s %s" % (src_filepath, dst_filepath))
        # shell.SHFileOperation((0, shellcon.FO_COPY, src_filepath, dst_filepath, shellcon.FOF_NOCONFIRMMKDIR, None, None))
        # copyfile(src_filepath, dst_filepath)

    print("finish merge gbuffer")

    # # WarpRes
    # for filePath in glob.glob(globalInfo.mResPath + "\\*"):
    #     filename = filePath.split('\\')[-1]
    #     dst_filepath = os.path.join(globalInfo.mMergeWarpRes, filename)
    #     if os.path.exists(dst_filepath) is False:
    #         shell.SHFileOperation((0, shellcon.FO_COPY, filePath, dst_filepath, shellcon.FOF_NOCONFIRMMKDIR, None, None))
    #         # copyfile(filePath, dst_filepath)

    # print("finish merge WarpRes")

    # # No_hole_Res
    # for filePath in glob.glob(globalInfo.mNoHolePath + "\\*"):
    #     filename = filePath.split('\\')[-1]
    #     dst_filepath = os.path.join(globalInfo.mMergeNoHole, filename)
    #     if os.path.exists(dst_filepath) is False:
    #         shell.SHFileOperation((0, shellcon.FO_COPY, filePath, dst_filepath, shellcon.FOF_NOCONFIRMMKDIR, None, None))
    #         # copyfile(filePath, dst_filepath)

    # print("finish merge No_hole_Res")

    # # OCC_Res
    # for filePath in glob.glob(globalInfo.mOCCPath + "\\*"):
    #     filename = filePath.split('\\')[-1]
    #     dst_filepath = os.path.join(globalInfo.mMergeOCC, filename)
    #     if os.path.exists(dst_filepath) is False:
    #         shell.SHFileOperation((0, shellcon.FO_COPY, filePath, dst_filepath, shellcon.FOF_NOCONFIRMMKDIR, None, None))
    #         # copyfile(filePath, dst_filepath)

    # print("finish merge OCC_Res")

    # # GT
    # for filePath in glob.glob(globalInfo.mGTPath + "\\*"):
    #     filename = filePath.split('\\')[-1]
    #     dst_filepath = os.path.join(globalInfo.mMergeGT, filename)
    #     if os.path.exists(dst_filepath) is False:
    #         shell.SHFileOperation((0, shellcon.FO_COPY, filePath, dst_filepath, shellcon.FOF_NOCONFIRMMKDIR, None, None))
    #         # copyfile(filePath, GT)

    # print("finish merge OCC_Res")

def main(args):

    globalInfo, start, end = init(args[1])

    if check(globalInfo, start, end) is False:
        return

    demodulate_process_list = list()
    make_hole_process_list = list()
    num = int((end - start) / Thread_NUM)

    for i in range(Thread_NUM):
        sub_start = start + i * num
        sub_end = sub_start + num
        if i == Thread_NUM - 1:
            sub_end = end
        if ENABLE_DEMODULATE:
            demodulate_process_list.append(Process(target=demodulate, args=(i, globalInfo, sub_start, sub_end,)))
        if i == Thread_NUM - 1:
            sub_end -= 5
        make_hole_process_list.append(Process(target=make_hole, args=(i, globalInfo, sub_start, sub_end,)))

    if ENABLE_DEMODULATE:
        for sub_process in demodulate_process_list:
            sub_process.start()
        for sub_process in demodulate_process_list:
            sub_process.join()

    print("finish demodulate.")

    for sub_process in make_hole_process_list:
        sub_process.start()
    for sub_process in make_hole_process_list:
        sub_process.join()
    print("finish make hole.")

    # merge_files(globalInfo, start + WARP_NUM, end)


if __name__ == "__main__":
    main(sys.argv)


