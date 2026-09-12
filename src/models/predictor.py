import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import logging

logger = logging.getLogger(__name__)

class SVDPredictor:
    """
    A Matrix Factorization Predictor using TruncatedSVD.
    This model decomposes the user-item matrix into latent factors
    to predict missing ratings.
    """
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.model = TruncatedSVD(n_components=self.n_components, random_state=42)
        self.user_index = None
        self.item_index = None
        self.user_factors = None
        self.item_factors = None
        self.global_mean = 0

    def fit(self, ratings: pd.DataFrame):
        logger.info(f"Fitting SVD Predictor with {self.n_components} components...")

        # 1. Calculate Global Mean for centering
        self.global_mean = ratings['rating'].mean()

        # 2. Create User-Item Matrix and Center it
        # We subtract the global mean so that '0' represents the average rating
        pivot_df = ratings.pivot_table(index='user_id', columns='item_id', values='rating', aggfunc='mean')
        centered_df = pivot_df.fillna(self.global_mean) - self.global_mean

        self.user_index = pivot_df.index
        self.item_index = pivot_df.columns

        # 3. SVD Decomposition on Centered Data
        self.user_factors = self.model.fit_transform(centered_df)
        self.item_factors = self.model.components_

    def predict(self, user_id, item_id):
        """
        Predicts the rating for a specific user-item pair.
        """
        try:
            u_idx = self.user_index.get_loc(user_id)
            i_idx = self.item_index.get_loc(item_id)

            # Dot product of factors + adding back the global mean
            prediction = np.dot(self.user_factors[u_idx], self.item_factors[:, i_idx]) + self.global_mean

            # Clip prediction to realistic rating range (1 to 5)
            return np.clip(prediction, 1, 5)
        except KeyError:
            # Return global mean for unknown users/items (Cold Start)
            return self.global_mean

    def predict_batch(self, user_ids, item_ids):
        """
        Predicts ratings for a batch of user-item pairs.
        """
        predictions = []
        for u, i in zip(user_ids, item_ids):
            predictions.append(self.predict(u, i))
        return np.array(predictions)
