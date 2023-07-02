from common.config import argparser
from common.utils import *
from common.engine import train, inference
from dataset.dataAugmentation import *
import torchvision.transforms as Transforms

def main(cfg):
    if (cfg.modality == "audio"):
        from model.audioModel import getModel
        from dataset.audioDataset import getLoader

        cfg.ckptLoadPath = cfg.aCkptLoadPath
        cfg.audioTransform = AudioAugmentationHelper([
            (0.9, RandomCrop(3))
        ])
    elif (cfg.modality == "visual"):
        from model.visualModel import getModel
        from dataset.visualDataset import getLoader

        cfg.ckptLoadPath = cfg.vCkptLoadPath
        if (not cfg.eval):
            cfg.trainImgTransforms = Transforms.Compose([
                Transforms.ToTensor(),
                Transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                Transforms.RandomResizedCrop((224, 224), (0.36, 0.64))
            ])
            cfg.validImgTransforms = Transforms.Compose([
                Transforms.ToTensor(),
                Transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                Transforms.CenterCrop((112, 122)),
                Transforms.Resize((224, 224))
            ])
        else:
            cfg.testImgTransforms = Transforms.Compose([
                Transforms.ToTensor(),
                Transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
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
        model.eval()
        inference(cfg, model, testLoader, device)

if (__name__ == "__main__"):
    cfg = argparser.parse_args()
    main(cfg)
