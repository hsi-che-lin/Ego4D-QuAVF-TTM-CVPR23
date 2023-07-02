import argparse

argparser = argparse.ArgumentParser()

argparser.add_argument("--modality", type = str, default = "audio", help = "Modality (audio or visual) to train / inference")
argparser.add_argument("--eval", action = "store_true", help = "Running mode")
argparser.add_argument("--startFromCkpt", type = bool, default = True, help = "Start from a checkpoint or random initialized model")
argparser.add_argument("--seed", type = int, default = 42, help = "Random seed")

# training
argparser.add_argument("--lr", type = float, default = 1e-5, help = "Learning rate")
argparser.add_argument("--numEpoch", type = int, default = 1, help = "Maximum number of epochs")
argparser.add_argument("--gradientClip", type = float, default = 1, help = "Gradient clipping")
argparser.add_argument("--earlyStop", type = int, default = 100, help = "Maximum patience for early stopping")
argparser.add_argument("--updateEveryBatch", type = int, default = 1, help = "Number of backward passes per update")
argparser.add_argument("--saveEveryEpochs", type = int, default = 1, help = "Save a checkpoint every k epochs")

# model
argparser.add_argument("--aBackboneType", type = str, default = "small", help = "variant of Whisper backbone used in the audio branch")
argparser.add_argument("--aDim", type = int, default = 768, help = "Feature dimension used in the audio branch")
argparser.add_argument("--aNumHeads", type = int, default = 8, help = "Number of heads used in the self-attention modeules in the audio branch")
argparser.add_argument("--aNumLayers", type = int, default = 1, help = "Number of additional self-attention modeules in the audio branch")
argparser.add_argument("--aDropout", type = float, default = 0.25, help = "Dropout rate used in the audio branch")
argparser.add_argument("--aFreezeBackbone", type = bool, default = True, help = "Freeze the Whisper backbone in the audio branch")

argparser.add_argument("--vBackboneType", type = str, default = "resnet50", help = "Backbone used in the vision branch")
argparser.add_argument("--vDim", type = int, default = 512, help = "Feature dimension used in the vision branch")
argparser.add_argument("--vNumHeads", type = int, default = 8, help = "Number of heads used in the self-attention modeules in the vision branch")
argparser.add_argument("--vNumLayers", type = int, default = 1, help = "Number of additional self-attention modeules in the vision branch")
argparser.add_argument("--scoreBin", type = int, default = 20, help = "Dimension of one-hot vector to represent the level of quality score")
argparser.add_argument("--vDropout", type = float, default = 0, help = "Dropout rate used in the vision branch")
argparser.add_argument("--vFreezeBackbone", type = bool, default = True, help = "Freeze the backbone in the vision branch")

# dataloader
argparser.add_argument("--batchSizeForward", type = int, default = 32, help = "Batch size per forward pass")
argparser.add_argument("--numWorkers", type = int, default = 4, help = "Number of workers used in dataloader")
argparser.add_argument("--minLength", type = float, default = 0.1, help = "Miniumum length for audio training data")
argparser.add_argument("--featLength", type = int, default = 15, help = "Length of visual feature")
argparser.add_argument("--frameStride", type = int, default = 15, help = "Stride between input visual frames")
argparser.add_argument("--dataStride", type = int, default = 8, help = "Stride between each training sample")
argparser.add_argument("--useScore", type = bool, default = True, help = "Use quality score as input in visual branch")
argparser.add_argument("--scoreThreshold", type = float, default = 0.3, help = "Threshold for quality score filtering")

# path
argparser.add_argument("--tensorboardPath", type = str, default = "./tensorboard", help = "Directory for tensorboard")
argparser.add_argument("--aCkptLoadPath", type = str, default = "./checkpoints/best.ckpt", help = "Path to the checkpoint for audio branch")
argparser.add_argument("--vCkptLoadPath", type = str, default = "./checkpoints/best.ckpt", help = "Path to the checkpoint for visual branch")
argparser.add_argument("--ckptSavePath", type = str, default = "./checkpoints", help = "Directory to store checkpoints")
argparser.add_argument("--scorePath", type = str, default = "./data/qualityScore.json", help = "Path to quality scores of training data")
argparser.add_argument("--audioPath", type = str, default = "./data/audio", help = "Directory for audio files")
argparser.add_argument("--faceCropPath", type = str, default = "./data/faceCrops", help = "Directory to store face crops images")
argparser.add_argument("--ttmPath", type = str, default = "./data/ttm", help = "Directory for ttm label files")
argparser.add_argument("--trainSplitPath", type = str, default = "./data/split/train.txt", help = "Path to the text file storing IDs of training data")
argparser.add_argument("--validSplitPath", type = str, default = "./data/split/valid.txt", help = "Path to the text file storing IDs of validation data")
argparser.add_argument("--testDataPath", type = str, default = "./data/social_test/final_test_data", help = "Directory for test data")
argparser.add_argument("--testScorePath", type = str, default = "./data/social_test/qualityScoreTest.json", help = "Path to quality scores of training data")
argparser.add_argument("--aPred", type = str, default = "./aPred.json", help = "Path to audio prediction")
argparser.add_argument("--vPred", type = str, default = "./vPred.json", help = "Path to video prediction")
