import os
import json
import face_alignment
import cv2
import numpy as np
from tqdm import tqdm

def processBbox(origBbox, newBbox, W, H):
    y1 = int(origBbox[1] - newBbox[1])
    y2 = int(origBbox[3] - newBbox[1])
    x1 = int(origBbox[0] - newBbox[0])
    x2 = int(origBbox[2] - newBbox[0])
    
    r = (1.2 - 1) / 2
    x12 = x2 - x1
    y12 = y2 - y1
    
    x1 = int(max(x1 - r * x12, 0))
    y1 = int(max(y1 - r * y12, 0))
    x2 = int(min(x2 + r * x12, W))
    y2 = int(min(y2 + r * y12, H))
    
    return x1, y1, x2, y2


if (__name__ == "__main__"):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
    faceCropPath = "../../data/faceCrops"
    metaDataPath = "../../data/faceCrops/metaData.json"
    splitPath = "../../data/faceCrops/split.txt"
    confidenceScore = {}

    with open(splitPath, "r") as f:
        uids = [line.strip() for line in f.readlines()]

    with open(metaDataPath, "r") as f:
        metaData = json.load(f)

    for uid in tqdm(uids):
        confidenceScore[uid] = {}
        uidPath = os.path.join(faceCropPath, uid)
        pids = os.listdir(uidPath)

        for pid in pids:
            confidenceScore[uid][pid] = {}
            pidPath = os.path.join(uidPath, pid)
            fids = os.listdir(pidPath)

            for fid in tqdm(fids, leave = False):
                imgPath = os.path.join(pidPath, fid)
                fid = fid.split(".")[0]
                info = metaData[uid][pid][fid]
                origBbox = info["origBbox"]
                newBbox = info["newBbox"]
                W, H = info["adjustInfo"]["origShape"]
                pad = info["adjustInfo"]["pad"]
                
                img = cv2.imread(imgPath)
                unPad = img[pad[2]:pad[3], pad[0]:pad[1]]
                resize = cv2.resize(unPad, (W, H))
                x1, y1, x2, y2 = processBbox(origBbox, newBbox, W, H)
                orig = resize[y1:y2, x1:x2, ::-1]
                
                bbox = [np.array([0, 0, orig.shape[1], orig.shape[0]])]
                args = {
                    "image_or_path": orig,
                    "detected_faces": bbox,
                    "return_landmark_score": True
                }
                landmarks, score, _ = fa.get_landmarks_from_image(**args)
                confidenceScore[uid][pid][fid] = score[0].mean().item()

    with open("./confidenceScore.json", "w") as f:
        json.dump(confidenceScore, f)