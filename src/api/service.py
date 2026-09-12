import pandas as pd
import sqlite3
import logging
from pathlib import Path
from src.models.predictor import SVDPredictor
from src.ranking.reranker import MultiObjectiveReranker

logger = logging.getLogger(__name__)

class RecommenderService:
    """
    A singleton service that loads the model and reranker
    into memory for fast API responses. Now reads from SQLite.
    """
    def __init__(self):
        self.base_path = Path(r'C:\Users\DrX DIPAK\projects\multi-objective-recommender')
        self.db_path = self.base_path / 'data' / 'recommender.db'
        self.model = None
        self.reranker = None
        self.all_item_ids = None
        self._load_resources()

    def _get_db_connection(self):
        return sqlite3.connect(self.db_path)

    def _load_resources(self):
        try:
            logger.info("Initializing Recommender Service from Database...")

            with self._get_db_connection() as conn:
                # Load data using SQL queries
                ratings = pd.read_sql("SELECT * FROM ratings", conn)
                item_summary = pd.read_sql("SELECT * FROM item_summary", conn)

            # Initialize and fit SVD model
            self.model = SVDPredictor(n_components=20)
            self.model.fit(ratings)

            # Initialize Multi-Objective Reranker
            self.reranker = MultiObjectiveReranker(item_summary)
            self.all_item_ids = ratings['item_id'].unique()

            logger.info("✅ Recommender Service resources loaded from DB successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to load resources: {e}")
            raise e

    def get_recommendations(self, user_id: str, n=10, mode='multi_objective'):
        """
        Generates recommendations.
        Modes: 'naive' (just SVD) or 'multi_objective' (SVD + Reranker).
        """
        # 1. Get Naive SVD scores
        predictions = []
        for item_id in self.all_item_ids:
            score = self.model.predict(user_id, item_id)
            predictions.append((item_id, score))

        predictions.sort(key=lambda x: x[1], reverse=True)

        if mode == 'naive':
            return [item_id for item_id, score in predictions[:n]]

        # 2. Apply Multi-Objective Reranking
        # We take top 50 as candidates to give the reranker more "Hidden Gems" to choose from
        candidates = predictions[:50]
        return self.reranker.rerank(user_id, candidates, n=n)

