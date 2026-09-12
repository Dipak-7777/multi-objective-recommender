import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import logging

logger = logging.getLogger(__name__)

class PopularityBaseline:
    """
    A simple baseline that recommends the most popular items to everyone.
    """
    def __init__(self):
        self.top_items = None

    def fit(self, ratings: pd.DataFrame):
        logger.info("Fitting Popularity Baseline...")
        # Calculate total interactions per item
        pop_df = ratings.groupby('item_id').size().sort_values(ascending=False)
        self.top_items = pop_df.index.tolist()

    def predict(self, user_id, n=10):
        # Returns the top N most popular items
        return self.top_items[:n]

class UserAverageBaseline:
    """
    A baseline that predicts a user's average rating for any item.
    """
    def __init__(self):
        self.user_means = None

    def fit(self, ratings: pd.DataFrame):
        logger.info("Fitting User-Average Baseline...")
        self.user_means = ratings.groupby('user_id')['rating'].mean()

    def predict(self, user_id):
        # Return the average rating for the user, or global average if user is new
        return self.user_means.get(user_id, self.user_means.mean())

def calculate_rmse(actual, predicted):
    """
    Calculates Root Mean Squared Error.
    Lower is better.
    """
    return np.sqrt(mean_squared_error(actual, predicted))
