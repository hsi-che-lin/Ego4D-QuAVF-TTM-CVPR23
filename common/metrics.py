r"""Adapted from AVA ASD."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import numpy as np
import pandas as pd


def eq(a, b, tolerance = 1e-09):
    """Returns true if values are approximately equal."""
    return abs(a - b) <= tolerance


def load_csv(file, column_names):
    """Loads CSV from the filename or lines using given column names.
    Adds uid column.
    Args:
        file: Filename or list to load.
        column_names: A list of column names for the data.
    Returns:
        df: A Pandas DataFrame containing the data.
    """
    # Here and elsewhere, df indicates a DataFrame variable.
    if isinstance(file, str):
        df = pd.read_csv(file, names = column_names)
    else:
        df = pd.DataFrame(file, columns = column_names)
    
    return df


def merge_groundtruth_and_predictions(df_groundtruth, df_predictions):
    """Merges groundtruth and prediction DataFrames.
    The returned DataFrame is merged on uid field and sorted in descending order
    by score field. Bounding boxes are checked to make sure they match between
    groundtruth and predictions.
    Args:
        df_groundtruth: A DataFrame with groundtruth data.
        df_predictions: A DataFrame with predictions data.
    Returns:
        df_merged: A merged DataFrame, with rows matched on uid column.
    """
    if df_groundtruth["uid"].count() != df_predictions["uid"].count():
        raise ValueError(
            "Groundtruth and predictions CSV must have the same number of "
            "unique rows.")

    if df_predictions["label"].unique() != [1]:
        raise ValueError(
            "Predictions CSV must contain only SPEAKING_AUDIBLE label.")

    if df_predictions["score"].count() < df_predictions["uid"].count():
        raise ValueError("Predictions CSV must contain score for every row.")

    # Merges groundtruth and predictions on uid, validates that uid is unique
    # in both frames, and sorts the resulting frame by the predictions score.
    tmp = df_groundtruth.merge(
        df_predictions,
        on = "uid",
        suffixes = ("_groundtruth", "_prediction"),
        validate = "1:1")
    df_merged = tmp.sort_values(by = ["score"], ascending = False).reset_index()

    return df_merged


def get_all_positives(df_merged):
    """Counts all positive examples in the groundtruth dataset."""
    return df_merged[df_merged["label_groundtruth"] == 1]["uid"].count()


def compute_average_precision(precision, recall):
    """Compute Average Precision according to the definition in VOCdevkit.
    Precision is modified to ensure that it does not decrease as recall
    decrease.
    Args:
        precision: A float [N, 1] numpy array of precisions
        recall: A float [N, 1] numpy array of recalls
    Raises:
        ValueError: if the input is not of the correct format
    Returns:
        average_precison: The area under the precision recall curve. NaN if
        precision and recall are None.
    """
    if precision is None:
        if recall is not None:
            raise ValueError("If precision is None, recall must also be None")
        return np.NAN

    if not isinstance(precision, np.ndarray) or not isinstance(
        recall, np.ndarray):
        raise ValueError("precision and recall must be numpy array")
    if precision.dtype != np.float or recall.dtype != np.float:
        raise ValueError("input must be float numpy array.")
    if len(precision) != len(recall):
        raise ValueError("precision and recall must be of the same size.")
    if not precision.size:
        return 0.0
    if np.amin(precision) < 0 or np.amax(precision) > 1:
        raise ValueError("Precision must be in the range of [0, 1].")
    if np.amin(recall) < 0 or np.amax(recall) > 1:
        raise ValueError("recall must be in the range of [0, 1].")
    if not all(recall[i] <= recall[i + 1] for i in range(len(recall) - 1)):
        raise ValueError("recall must be a non-decreasing array")

    recall = np.concatenate([[0], recall, [1]])
    precision = np.concatenate([[0], precision, [0]])

    # Smooth precision to be monotonically decreasing.
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = np.maximum(precision[i], precision[i + 1])

    indices = np.where(recall[1:] != recall[:-1])[0] + 1
    average_precision = np.sum(
        (recall[indices] - recall[indices - 1]) * precision[indices])
        
    return average_precision


def calculate_precision_recall(df_merged):
    """Calculates precision and recall going through df_merged row-wise."""
    all_positives = get_all_positives(df_merged)

    # Populates each row with 1 if this row is a true positive
    # (at its score level).
    df_merged["is_tp"] = np.where(
        (df_merged["label_groundtruth"] == 1) &
        (df_merged["label_prediction"] == 1), 1, 0)

    # Counts true positives up to and including that row.
    df_merged["tp"] = df_merged["is_tp"].cumsum()

    # Calculates precision for every row counting true positives up to
    # and including that row over the index (1-based) of that row.
    df_merged["precision"] = df_merged["tp"] / (df_merged.index + 1)

    # Calculates recall for every row counting true positives up to
    # and including that row over all positives in the groundtruth dataset.
    df_merged["recall"] = df_merged["tp"] / all_positives

    return np.array(df_merged["precision"]), np.array(df_merged["recall"])


def run_evaluation(groundtruth, predictions):
    """Runs Social evaluation, printing average precision result."""
    df_groundtruth = load_csv(groundtruth, ["uid", "label"])
    df_predictions = load_csv(predictions, ["uid", "label", "score"])
        
    APs = []
    for i in range(2):
        df_gt = copy.copy(df_groundtruth)
        df_pred = copy.copy(df_predictions)
        if i == 0:
            df_gt['label'] = 1 - df_gt['label']
            df_pred['score'] = 1 - df_pred['score']
        df_merged = merge_groundtruth_and_predictions(df_gt, df_pred)
        precision, recall = calculate_precision_recall(df_merged)
        AP = compute_average_precision(precision, recall)
        APs.append(AP)

    mAP = np.mean(APs)

    return mAP
