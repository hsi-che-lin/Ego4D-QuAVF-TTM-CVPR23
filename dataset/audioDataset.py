import os
import torch
import json
import whisper
import soundfile as sf
from random import randint
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

class TTMDataset(Dataset):
    def __init__(self, mode, cfg):
        self.mode = mode
        self.audTran = cfg.audioTransform
        self.minLength = int(cfg.minLength * 30)
        self.audioPath = cfg.audioPath
        getDataArg = dict(
            ttmPath = cfg.ttmPath,
            splitPath = getattr(cfg, f"{mode}SplitPath"),
        )
        self.dataList, self.pos, self.neg = self._getData(**getDataArg)
    

    def __len__(self):
        return len(self.dataList)


    def __getitem__(self, idx):
        data = self.dataList[idx]
        mel = self._getAudio(data)
        label = data["label"]
        data = {
            "mel": mel,
            "length": data["length"]
        }

        return data, label


    def _getData(self, ttmPath, splitPath):
        jsons = os.listdir(ttmPath)
        dataList = []
        wavLengths = {}
        pos = 0
        neg = 0
        progressBarArg = {
            "iterable": jsons,
            "desc": f"[Data] Preparing {self.mode} data",
            "total": len(jsons),
            "leave": False,
        }
        progressBar = tqdm(**progressBarArg)

        with open(splitPath, "r") as f:
            uids = [uid.strip() for uid in f.readlines()]

        for jsonFile in progressBar:
            uid = jsonFile.replace(".json", "") 
            
            if (uid not in uids): continue

            path = os.path.join(ttmPath, jsonFile)

            with open(path, "r") as f:
                ttms = json.load(f)

            for ttm in ttms:
                if (self._check(ttm, wavLengths)):
                    label = 1 if (ttm["target"] != None) else 0
                    start = int(ttm["startFrame"] / 30 * 16000)
                    end = int(ttm["endFrame"] / 30 * 16000)
                    length = ttm["endFrame"] - ttm["startFrame"]
                    dataList.append({
                        "uid": uid,
                        "label": label,
                        "start": start,
                        "end": end,
                        "length": length
                    })
                    
                    if (label):
                        pos += length
                    else:
                        neg += length

        print(f"[DATA] number of {self.mode} data = {len(dataList)}"
              f" (positive:negative = {pos}:{neg})")

        return dataList, pos, neg


    def _check(self, ttm, wavLengths):
        personCheck = ttm["person"] != 0
        lengthCheck = (ttm["endFrame"] - ttm["startFrame"]) > self.minLength
        
        if (ttm["uid"] not in wavLengths):
            audFile = os.path.join(self.audioPath, f'{ttm["uid"]}.wav')
            track = sf.SoundFile(audFile)
            wavLengths[ttm["uid"]] = track.frames
            track.close()
            
        wavCheck = int(ttm["endFrame"] / 30 * 16000) < wavLengths[ttm["uid"]]
        check = personCheck and lengthCheck and wavCheck

        return check

    def _getAudio(self, data):
        uid = data["uid"]
        length = data["length"]
        audFile = os.path.join(self.audioPath, f"{uid}.wav")
        track = sf.SoundFile(audFile)
        
        if (length > 30 * 30):
            toRead = 30 * 16000
            start = randint(data["start"], data["end"] - toRead)
        else:
            start = data["start"]
            end = data["end"]
            toRead = end - start
        
        track.seek(start)
        audio = torch.tensor(track.read(toRead), dtype = torch.float32)
        track.close()
        
        if (self.audTran != None):
            audio = self.audTran(audio)
            
        data["length"] = int(audio.shape[0] / 16000 * 30)
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        return mel
    
    def collater(self, batch):
        feat = [b[0]["mel"] for b in batch]

        feat = torch.stack(feat, dim = 0)
        labelTensor = torch.full((len(batch), 900), -100, dtype = torch.int64)

        for (i, b) in enumerate(batch):
            length = b[0]["length"]
            label = b[1]
            labelTensor[i][:length] = label
            
        return feat, labelTensor


class TTMTestDataset(Dataset):
    def __init__(self, dataPath):
        self.dataPath = dataPath
        self.dataList = self._getData()
    

    def __len__(self):
        return len(self.dataList)


    def __getitem__(self, idx):
        data = self.dataList[idx]
        mel = self._getAudio(data)
        info = {
            "sid": data["sid"],
            "fid2pred": data["fid2pred"],
            "startIdx": data["startIdx"]
        }

        return mel, info


    def _getData(self):
        sids = os.listdir(self.dataPath)
        dataList = []
        totalFid = 0
        progressBarArg = {
            "iterable": sids,
            "desc": f"[Data] Preparing testing data",
            "total": len(sids),
            "leave": False,
        }
        progressBar = tqdm(**progressBarArg)

        for sid in progressBar:
            fidPath = os.path.join(self.dataPath, sid, "face")
            fids = os.listdir(fidPath)
            fids = sorted([int(fid.split(".")[0]) for fid in fids])
            totalFid += len(fids)
            seg = (fids[-1] + 1) // (30 * 30)
            start = 0

            for _ in range(seg):
                end = start + (30 * 30)
                fid2pred = [
                    fid if (fid in fids) else None for fid in range(start, end)
                ]
                dataList.append({
                    "sid": sid,
                    "start": start,
                    "end": end,
                    "fid2pred": fid2pred,
                    "startIdx": 0
                })
                start = end
            
            if (start != (fids[-1] + 1)):
                prevStart = start
                start = max(0, fids[-1] - (30 * 30) + 1)
                end = fids[-1] + 1
                fid2pred = [
                    fid if (fid in fids) else None for fid in range(start, end)
                ]
                dataList.append({
                    "sid": sid,
                    "start": start,
                    "end": end,
                    "fid2pred": fid2pred,
                    "startIdx": (prevStart - start)
                })

        print(f"[DATA] number of target fids = {totalFid}")

        return dataList


    def _getAudio(self, data):
        sid = data["sid"]
        start = int(data["start"] / 30 * 16000)
        end = int(data["end"] / 30 * 16000)
        toRead = end - start
        audFile = os.path.join(self.dataPath, sid, "audio", "aud.wav")
        track = sf.SoundFile(audFile)
        totalFrame = track.frames
        diff = max(0, end - totalFrame)
        start = max(0, start - diff)
        
        track.seek(start)
        audio = torch.tensor(track.read(toRead), dtype = torch.float32)
        track.close()
        
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio)

        return mel
    

    def collater(self, datas):
        mels = torch.stack([d[0] for d in datas], dim = 0)
        infos = [d[1] for d in datas]

        return mels, infos


def getLoader(cfg, mode):
    if (mode != "test"):
        dataset = TTMDataset(mode = mode, cfg = cfg)

        loader = DataLoader(
            dataset     = dataset,
            batch_size  = cfg.batchSizeForward,
            shuffle     = (mode == "train"),
            num_workers = cfg.numWorkers,
            collate_fn  = dataset.collater,
            pin_memory  = True
        )
    else:
        dataset = TTMTestDataset(dataPath = cfg.testDataPath)

        loader = DataLoader(
            dataset     = dataset,
            batch_size  = cfg.batchSizeForward,
            shuffle     = False,
            num_workers = cfg.numWorkers,
            collate_fn  = dataset.collater,
            pin_memory  = True
        )

    return loader


if (__name__ == "__main__"):
    from dataAugmentation import *
    class config:
        def __init__(self):
            self.dataPaths = {
                "ttmPath": "../../../data/ttm",
                "audioPath": "../../../data/wave",
                "trainSplitPath": "../../../data/split/train.txt",
                "validSplitPath": "../../../data/split/valid.txt",
            }
            self.minLength = 0.1
            self.batchSizeForward = 4
            self.numWorkers = 2
            self.audioTransform = AudioAugmentationHelper([
                (0.9, AddAudNoise(3, 20))
            ])

    cfg = config()

    loader = getLoader(cfg, "valid")

    for (feat, label) in loader:
        print(feat.shape)
        print(label.shape)
        break