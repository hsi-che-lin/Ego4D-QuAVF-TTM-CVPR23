import torch

from random import random, randint


class AudioAugmentationHelper:
    def __init__(self, funcs):
        self.funcs = funcs


    def __call__(self, aud):
        out = aud
        
        for (p, func) in self.funcs:
            if (random() < p):
                out = func(out)
        
        return out


class AddAudNoise:
    def __init__(self, lowSNR, highSNR):
        assert (lowSNR < highSNR), f"lowSNR ({lowSNR}) > highSNR ({highSNR})"
        self.low = lowSNR
        self.high = highSNR


    def __call__(self, aud):
        targSNR = random() * (self.high - self.low) + self.low
        noise = torch.randn_like(aud)
        energyS = torch.linalg.vector_norm(aud, ord = 2, dim = -1) ** 2
        energyN = torch.linalg.vector_norm(noise, ord = 2, dim = -1) ** 2
        origSNR = 10 * (torch.log10(energyS) - torch.log10(energyN))
        scale = 10 ** ((origSNR - targSNR) / 20.0)
        scaledNoise = scale.unsqueeze(-1) * noise
        noisyAud = aud + scaledNoise

        return noisyAud


class RandomCrop:
    def __init__(self, minLength):
        self.minLength = minLength * 16000
    
    def __call__(self, aud):
        length = aud.shape[0]

        if (length < self.minLength):
            return aud
        
        tarLen = randint(self.minLength, length)
        start = randint(0, length - tarLen)
        audCrop = aud[start:(start + tarLen)]

        return audCrop
