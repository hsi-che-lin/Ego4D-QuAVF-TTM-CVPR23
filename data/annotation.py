import argparse
import os
import json

from tqdm import tqdm
from scipy.interpolate import interp1d


def merge_json(path_results, path_merges):
    """ merge av_train.json and av_val.json"""
    
    d = {}
    with open(path_merges, "w+", encoding="utf-8") as f0:
        for file in sorted(os.listdir(path_results)):
            if file == 'av_train.json':
                with open(os.path.join(path_results, file), "r", encoding="utf-8") as f1:
                    json_dict = json.load(f1)
                    d = json_dict.copy()
            elif file == 'av_val.json':
                with open(os.path.join(path_results, file), "r", encoding="utf-8") as f2:
                    json_dict = json.load(f2)
                    vs = json_dict['videos']
                    for v in vs:
                        d['videos'].append(v)
        json.dump(d, f0)


def getTTMResult(jsonPath, resultPath):
    """ extract ttm label from train_valid_merged.json """

    if (not os.path.exists(resultPath)):
        os.makedirs(resultPath)

    with open(jsonPath) as f:
        data = json.load(f)
    
    for video in tqdm(data["videos"], desc = "[ExtractTTM]", leave = False):
        for clip in video["clips"]:
            ttmList = []
            uid = clip["clip_uid"]
            segments = clip["social_segments_talking"]

            for segment in segments:
                if (segment["person"] == None): continue

                person = int(segment["person"].replace("'", ""))
                startFrame = segment["start_frame"]
                endFrame = segment["end_frame"]

                tmp = dict(
                    uid = uid,
                    person = person,
                    target = segment["target"],
                    startFrame = startFrame,
                    endFrame = endFrame
                )
                ttmList.append(tmp)

            jsonName = os.path.join(resultPath, uid + ".json")
            with open(jsonName, "a") as jsonF:
                json.dump(ttmList, jsonF)


def get_json(json_path, result_path):
    """ extract bounding box label from train_valid_merged.json """
    
    with open(json_path) as f:
        data = json.load(f)
        
    for video in tqdm(data['videos'], desc = "[ExtractBbox]", leave = False):
        for clip in video['clips']:
            uid = clip['clip_uid']
            person_data = clip['persons']
            
            for i in range(1, len(person_data)):                                # ignore camera wearer
                tracks = person_data[i]['tracking_paths']
                for track in tracks:
                    track_list = []
                    track_id = track['track_id']
                    track_data = track['track']
                    for td in track_data:
                        data = {
                            'clip_uid': uid,
                            'frameNumber': td['frame'],
                            'Person ID': person_data[i]['person_id'],
                            'x': td['x'],
                            'y': td['y'],
                            'height': td['height'],
                            'width': td['width']
                        }
                        track_list.append(data)

                    folder_path = os.path.join(result_path, uid)
                    json_name = track_id + '.json'
                    jsonfile_path = os.path.join(folder_path, json_name)
                    
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)
                        
                    with open(jsonfile_path, 'a') as json_file:
                        json.dump(track_list, json_file)


def mergeTrack(trackPath, resultPath):
    if (not os.path.exists(resultPath)):
        os.makedirs(resultPath)
        
    clips = os.listdir(trackPath)

    for clip in tqdm(clips, desc = "[mergeTrack]", leave = False):
        jsons = os.listdir(os.path.join(trackPath, clip))
        persons = {}

        for jsonFile in jsons:
            jsonPath = os.path.join(trackPath, clip, jsonFile)

            try:
                with open(jsonPath, "r") as f:
                    track = json.load(f)
            except:
                print(jsonPath)
                exit()
            
            bbox = [[], [], [], []]
            frame = []
            personID = track[0]["Person ID"]

            for data in track:
                x = float(data["x"])
                y = float(data["y"])
                w = float(data["width"])
                h = float(data["height"])

                if ((w < 0) or (h < 0) or (data["Person ID"] != personID)):
                    continue

                bbox[0].append(max(x, 0))
                bbox[1].append(max(y, 0))
                bbox[2].append(max(x, 0) + w)
                bbox[3].append(max(y, 0) + h)
                frame.append(int(data["frameNumber"]))

            completeFrame = list(range(min(frame), max(frame), 1))

            if (len(frame) != len(completeFrame)) and (len(frame) > 1):
                for i in range(4):
                    bbox[i] = interp1d(frame, bbox[i])(completeFrame)

            if (not personID in persons):
                persons[personID] = {}

            for (f, x1, y1, x2, y2) in zip(completeFrame, *bbox):
                persons[personID][f] = [x1, y1, x2, y2]
        
        resultJson = os.path.join(resultPath, f"{clip}.json")

        with open(resultJson, "w") as f:
            json.dump(persons, f)


if (__name__ == "__main__"):
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--avdPath", type = str, default = "./annotations", help = "Directory contains annoation files for Ego4D AV Diarization tasks")
    argparser.add_argument("--mergePath", type = str, default = "./annotations/train_valid_merged.json", help = "Path to the merged anootation file")
    argparser.add_argument("--ttmPath", type = str, default = "./ttm", help = "Directory to store the ttm label files")
    argparser.add_argument("--trackingPath", type = str, default = "./trackPath", help = "Directory to store the bounding box label files")
    arg = argparser.parse_args()
    trackTmp = "./trackTmp"

    print("Merging train validation annotation files...")
    merge_json(arg.avdPath, arg.mergePath)
    print("Extracting ttm labels...")
    getTTMResult(arg.mergePath, arg.ttmPath)
    print("Extracting bounding box labels...")
    get_json(arg.mergePath, trackTmp)
    mergeTrack(trackTmp, arg.trackingPath)
    os.system(f"rm -r {trackTmp}")
