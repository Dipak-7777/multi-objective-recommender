import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.models.predictor import SVDPredictor
from src.ranking.reranker import MultiObjectiveReranker

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # ABSOLUTE PATHS
    base_path = Path(r'C:\Users\DrX DIPAK\projects\multi-objective-recommender')
    processed_path = base_path / 'data' / 'processed'

    try:
        logger.info("Loading data for re-ranking...")
        ratings = pd.read_csv(processed_path / 'ratings_processed.csv')
        item_summary = pd.read_csv(processed_path / 'item_summary.csv')

        # 1. Initialize and Fit the SVD Predictor
        model = SVDPredictor(n_components=20)
        model.fit(ratings)

        # 2. Initialize the Multi-Objective Reranker
        reranker = MultiObjectiveReranker(item_summary)

        # 3. Test for a specific user
        test_user = ratings['user_id'].iloc[0]
        all_item_ids = ratings['item_id'].unique()

        # Step A: Get "Naive" SVD Recommendations
        predictions = []
        for item_id in all_item_ids:
            score = model.predict(test_user, item_id)
            predictions.append((item_id, score))

        # Sort by score descending to get top 20 candidates
        predictions.sort(key=lambda x: x[1], reverse=True)
        top_20_candidates = predictions[:20]

        naive_recs = [item_id for item_id, score in top_20_candidates]

        # Step B: Apply Multi-Objective Re-ranking
        reranked_recs = reranker.rerank(test_user, top_20_candidates, n=10)

        logger.info("\n" + "="*50)
        logger.info(f"RE-RANKING RESULTS FOR USER: {test_user}")
        logger.info("="*50)
        logger.info(f"Naive SVD Recs (Top 10): {naive_recs[:10]}")
        logger.info(f"Multi-Objective Recs (Top 10): {reranked_recs}")
        logger.info("="*50)

        # Analysis of difference
        overlap = set(naive_recs[:10]) & set(reranked_recs)
        logger.info(f"Overlap between Naive and Multi-Objective: {len(overlap)}/10")
        logger.info("If overlap is low, the reranker successfully introduced Diversity and Novelty!")

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise e

if __name__ == "__main__":
    main()
