from common.config import argparser
from common.utils import *
from common.engine import train, inference
from dataset.dataAugmentation import *
import torchvision.transforms as Transforms

def main(cfg):
    if (cfg.modality == "audio"):
        from model.audioModel import getModel
        from dataset.audioDataset import getLoader

        cfg.audioTransform = AudioAugmentationHelper([
            (0.9, RandomCrop(3))
        ])
    elif (cfg.modality == "visual"):
        from model.visualModel import getModel
        from dataset.visualDataset import getLoader

        if (not cfg.eval):
            cfg.trainImgTransforms = Transforms.Compose([
                Transforms.ToTensor()
            ])
            cfg.validImgTransforms = Transforms.Compose([
                Transforms.ToTensor()
            ])
        else:
            cfg.testImgTransforms = Transforms.Compose([
                Transforms.ToTensor()
            ])
    else:
        print('--modality should be either "audio" or "visual"')
    
    fixSeeds(cfg.seed)
    device = getDefaultDevice()
    model = getModel(cfg, device)
    
    if (not cfg.eval):
        trainLoader = getLoader(cfg, "train")
        validLoader = getLoader(cfg, "valid")
        train(cfg, model, trainLoader, validLoader, device)
    else:
        testLoader = getLoader(cfg, "test")
        freezeWeights(model)
        inference(cfg, model, testLoader, device)

if (__name__ == "__main__"):
    cfg = argparser.parse_args()
    main(cfg)
