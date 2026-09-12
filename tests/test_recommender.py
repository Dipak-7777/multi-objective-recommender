import pytest
import pandas as pd
import numpy as np
from src.models.predictor import SVDPredictor
from src.ranking.reranker import MultiObjectiveReranker

def test_svd_predictor_range():
    """
    Test that the SVD predictor always returns ratings within the 1-5 range.
    """
    # Create a tiny dummy dataset
    data = {
        'user_id': ['u1', 'u1', 'u2', 'u2'],
        'item_id': ['i1', 'i2', 'i1', 'i2'],
        'rating': [5, 1, 4, 2]
    }
    df = pd.DataFrame(data)

    model = SVDPredictor(n_components=2)
    model.fit(df)

    # Predict for a known user/item
    pred = model.predict('u1', 'i1')
    assert 1.0 <= pred <= 5.0, f"Prediction {pred} is outside the valid range 1-5"

def test_reranker_diversity():
    """
    Test that the reranker actually changes the order of items
    when diversity weights are high.
    """
    # Dummy item summary: items from the same category
    data = {
        'item_id': ['i1', 'i2', 'i3'],
        'category': ['Electronics', 'Electronics', 'Books'],
        'total_interactions': [100, 90, 10] # i1 and i2 are popular, i3 is novel
    }
    item_summary = pd.DataFrame(data)

    # Scenario: i1 and i2 are most relevant, but i3 is diverse
    initial_recs = [
        ('i1', 5.0),
        ('i2', 4.9),
        ('i3', 3.0)
    ]

    # Reranker with high diversity/novelty weights
    reranker = MultiObjectiveReranker(item_summary, w_relevance=0.1, w_novelty=0.5, w_diversity=0.4)
    reranked = reranker.rerank('user_1', initial_recs, n=3)

    # With high novelty/diversity, i3 (the Book/Novel item) should move up
    assert reranked[0] != 'i1' or reranked[1] == 'i3', "Reranker failed to introduce diversity/novelty"

def test_svd_cold_start():
    """
    Test that the model handles unknown users by returning the global mean.
    """
    # TruncatedSVD requires at least 2 features (items) to function.
    data = {
        'user_id': ['u1'],
        'item_id': ['i1', 'i2'],
        'rating': [4.0, 3.0]
    }
    # We need to reshape the data because one user with multiple items
    # results in a matrix with only 1 row. SVD needs more than 1 feature.
    df = pd.DataFrame({
        'user_id': ['u1', 'u1'],
        'item_id': ['i1', 'i2'],
        'rating': [4.0, 3.0]
    })

    model = SVDPredictor(n_components=1)
    model.fit(df)

    # Predict for a user not in the training set
    pred = model.predict('unknown_user', 'i1')
    # Global mean of [4.0, 3.0] is 3.5
    assert pred == 3.5, f"Cold start should return global mean (3.5), got {pred}"
