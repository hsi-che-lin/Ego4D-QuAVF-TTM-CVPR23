from common.config import argparser
import json
import os


def getData(cfg):
    dataPath = cfg.testDataPath
    scorePath = cfg.qualityScore
    videoSpan = (cfg.featLength - 1) * cfg.frameStride + 1
    frameStride = cfg.frameStride
    
    sids = os.listdir(dataPath)
    scoreDict = {}
    half = (videoSpan - 1) // 2
    
    with open(scorePath, "r") as f:
        scoreInfo = json.load(f)
    
    for sid in sids:
        fids, scores, indices = getFID(dataPath, sid, scoreInfo[sid])
        fids = [None] * half + fids + [None] * half
        scores = [0] * half + scores + [0] * half
        
        for idx in indices:
            end = idx + videoSpan + 1
            fid = fids[(idx + half)]
            fidSeg = fids[idx:end:frameStride]
            score = sum(scores[idx:end:frameStride]) / len(fidSeg)

            if (str(sid) not in scoreDict):
                scoreDict[str(sid)] = {}
            
            scoreDict[str(sid)][str(fid)] = score
        
    return scoreDict


def getFID(dataPath, sid, scoreInfo):
    fids = []
    scores = []
    indices = []
    imgPath = os.path.join(dataPath, sid, "face")
    fid2pred = sorted([int(x.split(".")[0]) for x in os.listdir(imgPath)])
    start = fid2pred[0]
    end = fid2pred[-1]
    
    for fid in range(start, end + 1):
        if (fid in fid2pred):
            fids.append(fid)
            scores.append(scoreInfo[str(fid)])
            indices.append(len(fids) - 1)
        else:
            fids.append(None)
            scores.append(0)
    
    return fids, scores, indices


if (__name__ == "__main__"):
    cfg = argparser.parse_args()
    vDict = {}
    results = []
    scoreDict = getData(cfg)

    with open(cfg.vPred, "r") as f:
        vision = json.load(f)["results"]

    with open(cfg.aPred, "r") as f:
        audio = json.load(f)["results"]

    for v in vision:
        vid = str(v["video_id"])
        fid = str(v["frame_id"])
        
        if (vid not in vDict):
            vDict[vid] = {}
            
        vDict[vid][fid] = v["score"]
        
    for a in audio:
        vid = str(a["video_id"])
        fid = str(a["frame_id"])
        quality = scoreDict[vid][fid]
        score = (1 - quality) * a["score"] + quality * vDict[vid][fid]
            
        results.append({
            "video_id": vid,
            "frame_id": fid,
            "label": 1,
            "score": score
        })

    output = {
        "version": "1.0",
        "challenge": "ego4d_talking_to_me",
        "results": results
    }

    with open("QuAVF.json", "w") as f:
        json.dump(output, f)
