import os
import json
import face_alignment
import numpy as np
from tqdm import tqdm

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D)
bbox = [np.array([0, 0, 224, 224])]

dataPath = "../../data/social_test/"
sids = os.listdir(os.path.join(dataPath, "final_test_data"))

confidentScore = {}

for sid in tqdm(sids, leave = False):
    fidPath = os.path.join(dataPath, "final_test_data", sid, "face")
    fids = os.listdir(fidPath)
    confidentScore[sid] = {}

    for fid in tqdm(fids, leave = False):
        imgPath = os.path.join(fidPath, fid)
        fid = fid.split(".")[0]
        args = {
            "image_or_path": imgPath,
            "detected_faces": bbox,
            "return_landmark_score": True
        }
        landmarks, score, _ = fa.get_landmarks_from_image(**args)
        confidentScore[sid][fid] = score[0].mean().item()

with open(f"confidenceScoreTest.json", "w") as f:
    json.dump(confidentScore, f)
