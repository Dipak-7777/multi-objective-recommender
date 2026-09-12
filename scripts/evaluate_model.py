import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.model_selection import train_test_split
from src.models.predictor import SVDPredictor
from src.monitoring.evaluation import precision_at_k, recall_at_k, calculate_coverage, get_top_n_recommendations

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # ABSOLUTE PATHS
    base_path = Path(r'C:\Users\DrX DIPAK\projects\multi-objective-recommender')
    processed_path = base_path / 'data' / 'processed'

    try:
        logger.info("Loading processed ratings for evaluation...")
        ratings = pd.read_csv(processed_path / 'ratings_processed.csv')

        # 1. Train-Test Split (Crucial for fair evaluation)
        # We split the data so the model is tested on ratings it has NEVER seen.
        train_df, test_df = train_test_split(ratings, test_size=0.2, random_state=42)
        logger.info(f"Split data: {len(train_df)} training samples, {len(test_df)} test samples.")

        # 2. Train Model on Training Set
        model = SVDPredictor(n_components=20)
        model.fit(train_df)

        # 3. Evaluation over Test Set
        all_item_ids = ratings['item_id'].unique()
        test_users = test_df['user_id'].unique()

        precisions = []
        recalls = []
        all_user_recs = []

        logger.info(f"Evaluating over {len(test_users)} users...")

        # We evaluate a sample of users to keep it efficient
        eval_users = np.random.choice(test_users, size=min(100, len(test_users)), replace=False)

        for user_id in eval_users:
            # Actual items the user liked (rating >= 4)
            actual_liked = test_df[(test_df['user_id'] == user_id) & (test_df['rating'] >= 4)]['item_id'].tolist()

            # Model's top 10 recommendations
            recs = get_top_n_recommendations(model, user_id, all_item_ids, n=10)

            precisions.append(precision_at_k(actual_liked, recs))
            recalls.append(recall_at_k(actual_liked, recs))
            all_user_recs.append(recs)

        # 4. Final Metrics Calculation
        avg_precision = np.mean(precisions)
        avg_recall = np.mean(recalls)
        coverage = calculate_coverage(all_user_recs, len(all_item_ids))

        logger.info("\n" + "="*30)
        logger.info("FINAL MODEL METRICS")
        logger.info("="*30)
        logger.info(f"Precision@10: {avg_precision:.4f}")
        logger.info(f"Recall@10:    {avg_recall:.4f}")
        logger.info(f"Catalog Coverage: {coverage:.4f}")
        logger.info("="*30)

        if coverage < 0.1:
            logger.info("Interpretation: LOW COVERAGE. The model is only recommending a few popular items. This is a strong reason to move to the 'Innovation' phase (Diversity/Novelty).")
        else:
            logger.info("Interpretation: GOOD COVERAGE.")

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise e

if __name__ == "__main__":
    main()
