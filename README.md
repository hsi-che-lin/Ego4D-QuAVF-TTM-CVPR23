# Ego4D-QuAVF-TTM-CVPR23
This repo is a codebase for our submission to the Talking-to-Me Challenge at CVPR 2023. This challenge focuses on the identification of social interactions in egocentric videos. Specifically, given a video and audio segment containing tracked faces of interest, the objective is to determine whether the person in each frame is talking to the camera wearer.

## Introduction
We employ separate models to process the audio and image modalities so that we can make better use of the labels in the dataset. By disentangling the two modalities, the noise or quality variations in one branch won't affect the other branch. Moreover, we also take the input quality of vision branch into account by introduing a quality score for each sample. This quality score is utilized to filter out inappropriate training data, serves as supplementary information for our model, and is further used in our quality-aware audio-visual fusion (QuAVF) approach. Our top-performing models achieved an accuracy of **71.2%** on the validation set and **67.4%** on the test set. More detail can be found in our [technical report]() and [presentation slides]().

## Qualitative Results
|Example 1|Example 2|
:--------:|:--------:
[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/UDtGi8Dm_vE/0.jpg)](https://www.youtube.com/watch?v=UDtGi8Dm_vE)|[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/AKgF48-VGD0/0.jpg)](https://www.youtube.com/watch?v=AKgF48-VGD0)

## Environment
Building the environment by executing the following command
```
sudo apt-get install ffmpeg
pip install -r requirement.txt
```

## Data Preparation
Please follow the intruction [here](./data/README.md)

## Train
You can change hyperparameters in [common/config.py](./common/config.py) and run
```
bash train.sh
```
Note that this script will train one branch (audio or visual) at a time, and you can specify the modality by modifying the "--modality" argument in the configuration.

## Inference
You can use:
```
bash inference.sh
```
Note that this script will only inference one branch (audio or visual) at a time. After obtaining the results of both branches, you can execute the following script to get the quality-aware fusion prediction.
```
python common/QuAVF.py \
--aPred "path to prediction of audio branch" \
--vPred "path to prediction of visual branch" \
--qualityScore "path to quality score"
```

