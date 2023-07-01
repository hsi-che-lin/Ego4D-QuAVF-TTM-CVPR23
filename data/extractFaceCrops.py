import argparse
import os
import json
import cv2
import numpy as np
from tqdm import tqdm

def processCoord(bbox, height, width):
    x1, y1, x2, y2 = bbox
    r = (2 - 1) / 2
    x12 = x2 - x1
    y12 = y2 - y1
    
    x1 = int(max(x1 - r * x12, 0))
    y1 = int(max(y1 - r * y12, 0))
    x2 = int(min(x2 + r * x12, width))
    y2 = int(min(y2 + r * y12, height))
    
    return x1, y1, x2, y2


def resizeAndPad(img, bbox, height, width):
    x1, y1, x2, y2 = bbox
    H, W, C = img.shape
    adjustInfo = {
        "origShape": (W, H),
        "pad": [0, 224, 0, 224]         # (r, l, u, d)
    }

    if ((x1 == 0) and (W < H)):
        tarW = int(W * 224 / H)
        tmp = cv2.resize(img, (tarW, 224))
        pad = 224 - tarW
        img = np.zeros((224, 224, 3))
        img[:, pad:, :] = tmp

        adjustInfo["pad"][0] = pad
    elif ((x2 == width) and (W < H)):
        tarW = int(W * 224 / H)
        tmp = cv2.resize(img, (tarW, 224))
        pad = 224 - tarW
        img = np.zeros((224, 224, 3))
        img[:, :-pad, :] = tmp

        adjustInfo["pad"][1] = pad
    elif ((y1 == 0) and (H < W)):
        tarH = int(H * 224 / W)
        tmp = cv2.resize(img, (224, tarH))
        pad = 224 - tarH
        img = np.zeros((224, 224, 3))
        img[pad:, :, :] = tmp

        adjustInfo["pad"][2] = pad
    elif ((y2 == height) and (H < W)):
        tarH = int(H * 224 / W)
        tmp = cv2.resize(img, (224, tarH))
        pad = 224 - tarH
        img = np.zeros((224, 224, 3))
        img[:-pad, :, :] = tmp

        adjustInfo["pad"][3] = pad
    else:
        img = cv2.resize(img, (224, 224))
    
    return img, adjustInfo



if (__name__ == "__main__"):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--ttmPath", type = str, default = "./ttm", help = "Directory to store the ttm label files")
    argparser.add_argument("--trackingPath", type = str, default = "./trackPath", help = "Directory to store the bounding box label files")
    argparser.add_argument("--clipPath", type = str, default = "./clips", help = "Directory to store video files")
    argparser.add_argument("--faceCropPath", type = str, default = "./faceCrops", help = "Directory to store face crops images")
    args = argparser.parse_args()
    
    ttmList = []
    curUID = None
    curCAP = None
    curTrack = None
    curWidth = None
    curHeight = None
    metaData = {}
    saved = {}

    jsons = os.listdir(args.ttmPath)

    for j in jsons:
        with open(os.path.join(args.ttmPath, j), "r") as f:
            ttmList += json.load(f)

    for ttm in tqdm(ttmList, leave = False):
        uid = str(ttm["uid"])
        person = str(ttm["person"])
        start = int(ttm["startFrame"])
        end = int(ttm["endFrame"])
        startMSEC = start / 30 * 1000

        if (uid != curUID):
            curUID = uid
            curCAP = cv2.VideoCapture(os.path.join(args.clipPath, f"{uid}.mp4"))
            curHeight = int(curCAP.get(cv2.CAP_PROP_FRAME_HEIGHT))
            curWidth = int(curCAP.get(cv2.CAP_PROP_FRAME_WIDTH))
            
            with open(os.path.join(args.trackingPath, f"{uid}.json"), "r") as f:
                curTrack = json.load(f)
                
        savePath = os.path.join(args.faceCropPath, uid, person)
        curCAP.set(cv2.CAP_PROP_POS_MSEC, startMSEC)
        
        if (int(person) < 1): continue
        if (person not in curTrack): continue
        
        if (uid not in saved):
            saved[uid] = {}
            metaData[uid] = {}

        if (person not in saved[uid]):
            saved[uid][person] = []
            metaData[uid][person] = {}
            os.makedirs(savePath, exist_ok = True)
        
        for fid in range(start, end + 1):
            success, img = curCAP.read()
            
            if (not success): continue
            if (fid in saved[uid][person]): continue
            if (str(fid) not in curTrack[person]): continue

            saved[uid][person].append(fid)
            origBbox = curTrack[person][str(fid)]
            newBbox = processCoord(origBbox, curHeight, curWidth)
            x1, y1, x2, y2 = newBbox
            img = img[y1:y2, x1:x2]
            img, adjustInfo = resizeAndPad(img, newBbox, curHeight, curWidth)
            cv2.imwrite(os.path.join(savePath, f"{fid}.jpg"), img)
            
            metaData[uid][person][str(fid)] = {
                "adjustInfo": adjustInfo,
                "origBbox": origBbox,
                "newBbox": newBbox
            }

    with open(os.path.join(args.faceCropPath, "metaData.json"), "w") as f:
        json.dump(metaData, f)
