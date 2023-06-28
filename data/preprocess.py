from merge import merge_json
from get_ttm_result import getTTMResult
from get_json import get_json

if (__name__ == "__main__"):
    path = "../../data/annotations"
    mergePath = "../../data/annotations/train_valid_merged.json"
    trackingPath = "../../data/trackPath"
    ttmPath = "../../data/ttm"

    merge_json(path, mergePath)
    get_json(mergePath, trackingPath)
    getTTMResult(mergePath, ttmPath)

