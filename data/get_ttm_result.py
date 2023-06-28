import os
import json

def getTTMResult(jsonPath, resultPath):
    if (not os.path.exists(resultPath)):
        os.makedirs(resultPath)

    with open(jsonPath) as f:
        data = json.load(f)
    
    for video in data["videos"]:
        for clip in video["clips"]:
            ttmList = []
            uid = clip["clip_uid"]
            segments = clip["social_segments_talking"]

            print(uid)

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


# result_path
# - clip_uid 1.json
#   - ttm_list[0] = {"uid", "person", "target", "start_frame", "end_frame"}
#   - ttm_list[1] = ...
#   - ttm_list[2] = ...
#   - ...
# - clip_uid 2.json
# - clip_uid 3.json
# - ...
