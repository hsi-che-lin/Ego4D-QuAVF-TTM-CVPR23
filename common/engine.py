import os
import torch
import numpy as np
import json

from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import softmax
from tqdm import tqdm
from utils import saveCkpt, MetricsAccumulator, lossWeight
from metrics import run_evaluation


def trainStep(model, device, trainLoader, criterion, optim, updateRate,
              gradClip, trainWriter, epoch, endEpoch, trainIter, metricsDict):
    model.train()
    model.zero_grad()
    batchAccum = MetricsAccumulator(metricsDict)
    epochAccum = MetricsAccumulator(metricsDict)
    progressBarArg = {
        "iterable": enumerate(trainLoader, start = 1),
        "desc": f"[Train][{epoch}/{endEpoch}]",
        "total": len(trainLoader),
        "leave": False,
    }
    progressBar = tqdm(**progressBarArg)
    
    for (i, (data, label)) in progressBar:
        label = label.to(device)
        
        logits = model(data, device)
        loss = criterion(logits, label) / updateRate
        loss.backward()
        
        with torch.no_grad():
            pred = torch.argmax(logits, dim = 1)
            nData = (label != -100).sum().item()
            metricsDict["loss"] = loss.item() * nData * updateRate
            metricsDict["hits"] = (pred == label).sum().item()
            batchAccum.update(nData, metricsDict)
        
        if (((i % updateRate) == 0) or (i == len(trainLoader))):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm = gradClip)
            optim.step()
            model.zero_grad()
            
            avgDict = batchAccum.avg()
            trainWriter.add_scalar("Loss/batch", avgDict["loss"], trainIter)
            trainWriter.add_scalar("Acc/batch", avgDict["hits"], trainIter)
            
            epochAccum.update(batchAccum.nData, batchAccum.metrics)
            batchAccum.reset()
            trainIter += 1
    
    avgDict = epochAccum.avg()
    trainWriter.add_scalar("Loss/epoch", avgDict["loss"], epoch)
    trainWriter.add_scalar("Acc/epoch", avgDict["hits"], epoch)
    
    return trainIter


def validateStep(model, device, loader, criterion, writer, epoch, endEpoch,
                 metricsDict, sampleRatio, writeAccLoss):
    model.eval()
    validAccum = MetricsAccumulator(metricsDict)
    progressBarArg = {
        "iterable": enumerate(loader),
        "desc": f"[Valid][{epoch}/{endEpoch}]",
        "total": int(len(loader) * sampleRatio),
        "leave": False,
    }
    progressBar = tqdm(**progressBarArg)
    gt = open("gt.tmp.csv", "w")
    pr = open("pr.tmp.csv", "w")
    uid = 0
    
    for (i, (data, label)) in progressBar:
        label = label.to(device)
        
        with torch.no_grad():
            logits = model(data, device)
            pred = torch.argmax(logits, dim = 1)
            nData = (label != -100).sum()
            scores = softmax(logits, dim = 1)[:, 1]
            metricsDict["loss"] = criterion(logits, label).item() * nData
            metricsDict["hits"] = (pred == label).sum().item()
            validAccum.update(nData, metricsDict)
        
        for (s, l) in zip(scores.reshape(-1), label.reshape(-1)):
            if (l == -100): continue
            gt.write(f"{uid:0>8d},{l.item()}\n")
            pr.write(f"{uid:0>8d},{1},{s.item()}\n")
            uid += 1
        
        if (i == int(len(loader) * sampleRatio - 1)):
            break
    
    gt.close()
    pr.close()
    
    mAP = run_evaluation("gt.tmp.csv", "pr.tmp.csv")
    writer.add_scalar("mAP/epoch", mAP, epoch)
    
    os.remove("gt.tmp.csv")
    os.remove("pr.tmp.csv")
    
    if (writeAccLoss):
        avgDict = validAccum.avg()
        writer.add_scalar("Loss/epoch", avgDict["loss"], epoch)
        writer.add_scalar("Acc/epoch", avgDict["hits"], epoch)
    
    return mAP


def train(cfg, model, trainLoader, validLoader, device):
    numEpoch     = cfg.numEpoch
    gradClip     = cfg.gradientClip
    earlyStop    = cfg.earlyStop
    weight       = lossWeight(trainLoader, device)
    criterion    = torch.nn.CrossEntropyLoss(weight = weight)
    updateRate   = cfg.updateEveryBatch
    tbPath       = cfg.tensorboardPath
    savePath     = cfg.ckptSavePath
    trainWriter  = SummaryWriter(os.path.join(tbPath, "train"))
    validWriter  = SummaryWriter(os.path.join(tbPath, "valid"))
    metricsDict  = {"loss": 0, "hits": 0}
    patience     = 0
    bestmAP      = 0
    param        = filter(lambda p: p.requires_grad, model.parameters())
    optim        = torch.optim.Adam(param, lr = cfg.lr)
    
    if (cfg.startFromCkpt):
        try:
            optim.load_state_dict(torch.load(cfg.ckptLoadPath)["optim"])
        except:
            pass
    
    if (isinstance(cfg.saveEveryEpochs, int)):
        savePeriod = cfg.saveEveryEpochs
        saveCnt = 0
    else:
        savePeriod = -1
        saveCnt = 0
    
    for epoch in range(1, numEpoch + 1):
        # ------- Training -------
        model.train()
        trainArg = dict(
            model       = model,
            device      = device,
            trainLoader = trainLoader,
            criterion   = criterion,
            optim       = optim,
            updateRate  = updateRate,
            gradClip    = gradClip,
            trainWriter = trainWriter,
            epoch       = epoch,
            endEpoch    = numEpoch,
            trainIter   = trainIter,
            metricsDict = metricsDict
        )
        trainIter = trainStep(**trainArg)
        # --- End of Training ---
        
        # ------- Validation -------
        model.eval()
        validateArg = dict(
            model        = model,
            device       = device,
            loader       = trainLoader,
            criterion    = criterion,
            writer       = trainWriter,
            epoch        = epoch,
            endEpoch     = numEpoch,
            metricsDict  = metricsDict,
            sampleRatio  = 0.1,
            writeAccLoss = False
        )
        _ = validateStep(**validateArg)
        
        validateArg = dict(
            model        = model,
            device       = device,
            loader       = validLoader,
            criterion    = criterion,
            writer       = validWriter,
            epoch        = epoch,
            endEpoch     = numEpoch,
            metricsDict  = metricsDict,
            sampleRatio  = 1,
            writeAccLoss = True
        )
        mAP = validateStep(**validateArg)
        # --- End of Validation ---
        
        if (savePeriod != -1):
            saveCnt += 1
        
        if ((epoch == 1) or (epoch == numEpoch) or (saveCnt == savePeriod)):
            saveCnt %= savePeriod
            ckptPath = os.path.join(savePath, f"epoch{epoch}.ckpt")
            saveCkpt(ckptPath, bestmAP, model, optim, epoch, trainIter)
            
            print(f"Saving model at epoch {epoch}")

        if (mAP > bestmAP):
            bestmAP = mAP
            patience = 0
            ckptPath = os.path.join(savePath, "best.ckpt")
            saveCkpt(ckptPath, bestmAP, model, optim, epoch, trainIter)
            
            print(f"Saving model at epoch {epoch} (best)")
        else:
            patience += 1
            
            if (patience > earlyStop):
                print("model is not improving stop training")
                break


def audioInference(model, loader, device, outputPath):
    results = []
    progressBarArg = {
        "iterable": loader,
        "desc": "[Test]",
        "total": len(loader),
        "leave": False,
    }
    progressBar = tqdm(**progressBarArg)

    for (feat, infos) in progressBar:
        feat = feat.to(device)
        prevScore = 0.5

        with torch.no_grad():
            pred = model(feat, device)
            scores = softmax(pred, dim = 1)[:, 1]
        
        for (info, score) in zip(infos, scores):
            startIdx = info["startIdx"]
            fid2pred = info["fid2pred"][startIdx:]

            for (i, fid) in enumerate(fid2pred):
                if (fid == None): continue

                s = float(score[startIdx + i])
                
                if (np.isnan(s)):
                    s = prevScore
                else:
                    prevScore = s
                    
                results.append({
                    "video_id": info["sid"],
                    "frame_id": int(fid),
                    "label": 1,
                    "score": s
                })

        if (progressBar.total != len(progressBar.iterable)):
            progressBar.total = len(progressBar.iterable)
            progressBar.refresh()

    output = {
        "version": "1.0",
        "challenge": "ego4d_talking_to_me",
        "results": results
    }
    with open(outputPath, "w") as f:
        json.dump(output, f, indent = 4)


def visualInference(model, loader, device, outputPath):
    results = []
    progressBarArg = {
        "iterable": loader,
        "desc": "[Test]",
        "total": len(loader),
        "leave": False,
    }
    progressBar = tqdm(**progressBarArg)

    for (feat, info) in progressBar:
        prevScore = 0.5
        
        with torch.no_grad():
            pred = model(feat, device)
            scores = softmax(pred, dim = 1)[:, 1]
        
        for (sid, fid, score) in zip(info["sid"], info["fid"], scores):
            score = float(score.item())
            
            if (np.isnan(score)):
                score = prevScore
            else:
                prevScore = score
                
            results.append({
                "video_id": str(sid),
                "frame_id": int(fid.item()),
                "label": 1,
                "score": score
            })

        if (progressBar.total != len(progressBar.iterable)):
            progressBar.total = len(progressBar.iterable)
            progressBar.refresh()

    output = {
        "version": "1.0",
        "challenge": "ego4d_talking_to_me",
        "results": results
    }
    with open(outputPath, "w") as f:
        json.dump(output, f, indent = 4)


def inference(cfg, model, loader, device):
    if (cfg.modality == "audio"):
        audioInference(model, loader, device, cfg.aPred)
    elif (cfg.modality == "visual"):
        visualInference(model, loader, device, cfg.vPred)