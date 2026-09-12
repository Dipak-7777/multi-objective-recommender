import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def precision_at_k(actual, predicted, k=10):
    """
    Precision@K: Proportion of recommended items in the top-K that are relevant.
    """
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if not act_set:
        return 0.0
    return len(act_set & pred_set) / k

def recall_at_k(actual, predicted, k=10):
    """
    Recall@K: Proportion of relevant items that are found in the top-K.
    """
    act_set = set(actual)
    pred_set = set(predicted[:k])
    if not act_set:
        return 0.0
    return len(act_set & pred_set) / len(act_set)

def calculate_coverage(all_predictions, total_items):
    """
    Coverage: Proportion of the total item catalog recommended at least once.
    """
    unique_recs = set()
    for user_recs in all_predictions:
        unique_recs.update(user_recs)

    return len(unique_recs) / total_items

def get_top_n_recommendations(model, user_id, all_item_ids, n=10):
    """
    Generates top-N recommendations for a user.
    """
    predictions = []
    for item_id in all_item_ids:
        score = model.predict(user_id, item_id)
        predictions.append((item_id, score))

    # Sort by score descending
    predictions.sort(key=lambda x: x[1], reverse=True)
    return [item_id for item_id, score in predictions[:n]]
