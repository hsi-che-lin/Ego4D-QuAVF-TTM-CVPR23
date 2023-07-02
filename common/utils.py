import random
import numpy as np
import torch

def fixSeeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  

    return


def getDefaultDevice():
    if (torch.cuda.is_available()):
        return "cuda"
    else:
        return "cpu"


def countParam(model):
    total = sum([p.numel() for p in model.parameters()])
    trainable = sum([p.numel() for p in model.parameters() if p.requires_grad])

    return total, trainable


def freezeWeights(model):
    for p in model.parameters():
        p.requires_grad = False


def lossWeight(trainLoader, device):
    pos = trainLoader.dataset.pos
    neg = trainLoader.dataset.neg
    total = pos + neg
    weight = torch.tensor([(pos / total), (neg / total)], device = device)
    
    return weight


def saveCkpt(path, metric, model, optim, epoch, trainIter):
    torch.save({
        "metric": metric,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "epoch": epoch,
        "trainIter": trainIter
    }, path)

    return 


class MetricsAccumulator:
    def __init__(self, metricsNameList):
        self.nData = 0
        self.metrics = {
            name: 0 for name in metricsNameList
        }
    

    def update(self, nData, metrics):
        self.nData += nData

        for (k, v) in metrics.items():
            self.metrics[k] += v


    def avg(self):
        avgMetrics = {}

        for (k, v) in self.metrics.items():
            avgMetrics[k] = v / self.nData
            avgMetrics[k] = v / self.nData
        
        return avgMetrics
    

    def reset(self):
        self.nData = 0
        
        for k in self.metrics:
            self.metrics[k] = 0
