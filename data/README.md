# Data Preparation

Download the annoations and clips for audio-visual diarization benchmark following the Ego4D download [instructions](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md).

The structure should be like

data/
* annotations/
    * av_train.json
    * av_val.json
* split/
    * test.list
    * train.list
    * val.list
    * full.list
* clips/
    * 0b4cacb1-970f-4ef0-85da-371d81f899e0.mp4
    * 0c7d73eb-447f-4455-a4f6-f9d970ff9645.mp4
    * ...
* social_test/
    * final_test_data
        * 00a06cd8d7a0a5a9d9e72c918142048a
        * 00a79c41fcf979991ec12cf5ca8d920d
        * ...

## Annoations
Run the following script to extract the ttm labels and the bounding box labels
```
python annotation.py
```

## Video and Audio data
Run the following script to prepare the video and audio data
```
bash data.sh
```

## Quality Scores
Run the following script to prepare quality scores
```
python faceScore.py
python faceScoreTest.py
```
