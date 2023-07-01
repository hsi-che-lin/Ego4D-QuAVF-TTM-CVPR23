import argparse
import os
import json
import face_alignment
import numpy as np
from tqdm import tqdm

argparser = argparse.ArgumentParser()
argparser.add_argument("--dataPath", type = str, default = "./social_test/final_test_data", help = "Directory to store the test data")
argparser.add_argument("--resultPath", type = str, default = "./social_test/confidenceScoreTest.json", help = "Result file to store quality scores")
args = argparser.parse_args()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D)
bbox = [np.array([0, 0, 224, 224])]

sids = os.listdir(args.dataPath)

confidentScore = {}

for sid in tqdm(sids, leave = False):
    fidPath = os.path.join(args.dataPath, sid, "face")
    fids = os.listdir(fidPath)
    confidentScore[sid] = {}

    for fid in tqdm(fids, leave = False):
        imgPath = os.path.join(fidPath, fid)
        fid = fid.split(".")[0]
        faArgs = {
            "image_or_path": imgPath,
            "detected_faces": bbox,
            "return_landmark_score": True
        }
        landmarks, score, _ = fa.get_landmarks_from_image(**faArgs)
        confidentScore[sid][fid] = score[0].mean().item()

with open(args.resultPath, "w") as f:
    json.dump(confidentScore, f)
