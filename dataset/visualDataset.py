import os
import torch
import json
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

class TTMDataset(Dataset):
    def __init__(self, mode, cfg):
        assert ((cfg.featLength % 2) == 1), "expect odd feature length"
        
        self.imgPath = cfg.imgPath
        self.frameStride = cfg.frameStride
        self.imgTran = getattr(cfg, f"{mode}ImgTransforms")
        self.useScore = cfg.useScore
        getTTMArg = dict(
            mode = mode,
            cfg = cfg,
            videoSpan = (cfg.featLength - 1) * cfg.frameStride + 1,
            scoreThreshold = cfg.scoreThreshold,
            dataStride = cfg.dataStride
        )
        self.dataList, self.pos, self.neg = self._getData(**getTTMArg)


    def __len__(self):
        return len(self.dataList)


    def __getitem__(self, idx):
        data = self.dataList[idx]
        faceCrop = self._getImg(data)
        label = data["label"]
        
        if (self.useScore):
            feat = {
                "faceCrops": faceCrop,
                "score": data["score"]
            }
        else:
            feat = {"faceCrops": faceCrop}
        
        return feat, label


    def _getData(self, mode, cfg, videoSpan, scoreThreshold, dataStride):
        ttmPath = cfg.ttmPath
        scorePath = cfg.scorePath
        splitPath = getattr(cfg, f"{mode}SplitPath")
        jsons = os.listdir(ttmPath)
        dataList = []
        NegPos = [0, 0]
        half = (videoSpan - 1) // 2
        progressBarArg = {
            "iterable": jsons,
            "desc": f"[Data] Preparing {mode} data",
            "total": len(jsons),
            "leave": False,
        }
        progressBar = tqdm(**progressBarArg)
        
        with open(scorePath, "r") as f:
            scoreInfo = json.load(f)
        
        with open(splitPath, "r") as f:
            uids = [uid.strip() for uid in f.readlines()]
        
        for jsonFile in progressBar:
            if (jsonFile.replace(".json", "") not in uids): continue
            
            path = os.path.join(ttmPath, jsonFile)
            
            with open(path, "r") as f:
                ttms = json.load(f)
            
            for ttm in ttms:
                uid = ttm["uid"]
                person = str(ttm["person"])
                label = 1 if (ttm["target"] != None) else 0
                
                if (int(person) < 1): continue
                
                fids, scores = self._getFID(ttm, scoreInfo)
                fids = [None] * half + fids + [None] * half
                scores = [0] * half + scores + [0] * half
                idx = 0
                
                while (idx < (len(fids) - videoSpan - 1)):
                    end = idx + videoSpan + 1
                    fidSeg = fids[idx:end:self.frameStride]
                    score = sum(scores[idx:end:self.frameStride]) / len(fidSeg)
                    
                    if (score >= scoreThreshold):
                        dataList.append({
                            "uid": uid,
                            "person": person,
                            "label": label,
                            "fids": fidSeg,
                            "score": score
                        })
                        
                        NegPos[label] += 1
                        idx += dataStride
                    else:
                        idx += 1
        
        print(f"[DATA] number of {mode} data = {len(dataList)} "
              f"(Positive:Negative = {NegPos[1]}:{NegPos[0]})")
        
        return dataList, NegPos[1], NegPos[0]
    

    def _getFID(self, ttm, scoreInfo):
        fids = []
        scores = []
        uid = ttm["uid"]
        person = str(ttm["person"])
        start = ttm["startFrame"]
        end = ttm["endFrame"]
        imgPath = os.path.join(self.imgPath, uid, person)

        if ((uid in scoreInfo) and (person in scoreInfo[uid])):
            scoreInfo = scoreInfo[uid][person]
        else:
            return fids, scores

        
        for fid in range(start, end + 1):
            path = os.path.join(imgPath, f"{fid}.jpg")
            
            if (os.path.exists(path)):
                fids.append(fid)
                scores.append(scoreInfo[str(fid)])
            else:
                fids.append(None)
                scores.append(0)
        
        return fids, scores


    def _getImg(self, data):
        fcList = []
        uid = data["uid"]
        person = data["person"]
        fids = data["fids"]
        imgPath = os.path.join(self.imgPath, uid, person)
        
        for fid in fids:
            if (fid != None):
                path = os.path.join(imgPath, f"{fid}.jpg")
                img = Image.open(path)
                img = self.imgTran(img)
                fcList.append(img)
            else:
                fcList.append(torch.zeros((3, 224, 224)))
        
        faceCrop = torch.stack(fcList, dim = 0)
        
        return faceCrop


class TTMTestDataset(Dataset):
    def __init__(self, cfg, featLength, frameStride, useScore,
                 imgTransforms):
        assert ((featLength % 2) == 1), "expect odd feature length"
        
        self.dataPath = cfg.testDataPath
        self.frameStride = frameStride
        self.imgTran = imgTransforms
        self.useScore = useScore
        videoSpan = (featLength - 1) * frameStride + 1
        self.dataList = self._getData(cfg.testScorePath, videoSpan)


    def __len__(self):
        return len(self.dataList)


    def __getitem__(self, idx):
        data = self.dataList[idx]
        faceCrop = self._getImg(data)
        info = {
            "sid": data["sid"],
            "fid": data["fid"]
        }
        
        if (self.useScore):
            feat = {
                "faceCrops": faceCrop,
                "score": data["score"]
            }
        else:
            feat = {"faceCrops": faceCrop}
        
        return feat, info


    def _getData(self, scorePath, videoSpan):
        sids = os.listdir(self.dataPath)
        dataList = []
        half = (videoSpan - 1) // 2
        progressBarArg = {
            "iterable": sids,
            "desc": f"[Data] Preparing test data",
            "total": len(sids),
            "leave": False,
        }
        progressBar = tqdm(**progressBarArg)
        
        with open(scorePath, "r") as f:
            scoreInfo = json.load(f)
        
        for sid in progressBar:
            fids, scores, indices = self._getFID(sid, scoreInfo[sid])
            fids = [None] * half + fids + [None] * half
            scores = [0] * half + scores + [0] * half
            
            for idx in indices:
                end = idx + videoSpan + 1
                fid = fids[(idx + half)]
                fidSeg = fids[idx:end:self.frameStride]
                score = sum(scores[idx:end:self.frameStride]) / len(fidSeg)
                dataList.append({
                    "sid": sid,
                    "fid": fid,
                    "fids": fidSeg,
                    "score": score
                })
        
        print(f"[DATA] number of test data = {len(dataList)}")
        
        return dataList
    

    def _getFID(self, sid, scoreInfo):
        fids = []
        scores = []
        indices = []
        imgPath = os.path.join(self.dataPath, sid, "face")
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


    def _getImg(self, data):
        fcList = []
        sid = data["sid"]
        fids = data["fids"]
        imgPath = os.path.join(self.dataPath, sid, "face")
        
        for fid in fids:
            if (fid != None):
                path = os.path.join(imgPath, f"{fid}.jpg")
                img = Image.open(path)
                img = self.imgTran(img)
                fcList.append(img)
            else:
                fcList.append(torch.zeros((3, 224, 224)))
        
        faceCrop = torch.stack(fcList, dim = 0)
        
        return faceCrop


def getLoader(cfg, mode):
    if (mode != "test"):
        dataset = TTMDataset(mode = mode, cfg = cfg)

        loader = DataLoader(
            dataset     = dataset,
            batch_size  = cfg.batchSizeForward,
            shuffle     = (mode == "train"),
            num_workers = cfg.numWorkers,
            pin_memory  = True
        )
    else:
        dataset = TTMTestDataset(
            dataPaths      = cfg.dataPaths,
            featLength     = cfg.featLength,
            frameStride    = cfg.frameStride,
            useScore       = cfg.useScore,
            imgTransforms  = cfg.testImgTransforms
        )

        loader = DataLoader(
            dataset     = dataset,
            batch_size  = cfg.batchSizeForward,
            shuffle     = False,
            num_workers = cfg.numWorkers,
            pin_memory  = True
        )

    return loader


if (__name__ == "__main__"):
    from torchvision import transforms
    class config:
        def __init__(self):
            self.dataPaths = {
                "ttmPath": "../../../data/ttm",
                "imgPath": "../../../data/faceCrops",
                "scorePath": "../../../data/faceCrops/existScore.json",
                "trainSplitPath": "../../../data/split/train.txt",
                "validSplitPath": "../../../data/split/valid.txt"
            }
            self.featLength = 11
            self.frameStride = 2
            self.dataStride = 4
            self.scoreThreshold = 0.5
            self.useScore = True
            self.trainImgTransforms = transforms.Compose([
                transforms.ToTensor()
            ])
            self.validImgTransforms = transforms.Compose([
                transforms.ToTensor()
            ])
            self.batchSizeForward = 4
            self.numWorkers = 2

    cfg = config()

    loader = getLoader(cfg, "valid")

    for (feat, label) in loader:
        print(label.shape)
        print(feat["faceCrops"].shape)
        print(feat["score"].shape)
        break