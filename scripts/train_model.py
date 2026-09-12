import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.models.predictor import SVDPredictor
from src.recommenders.baselines import calculate_rmse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # ABSOLUTE PATHS for stability
    base_path = Path(r'C:\Users\DrX DIPAK\projects\multi-objective-recommender')
    processed_path = base_path / 'data' / 'processed'

    try:
        logger.info("Loading processed ratings...")
        ratings = pd.read_csv(processed_path / 'ratings_processed.csv')

        # --- ML MODEL: SVD Predictor ---
        model = SVDPredictor(n_components=20)
        model.fit(ratings)

        # Evaluate on a sample of the dataset
        # We use a sample to keep the evaluation fast and robust
        sample_size = min(2000, len(ratings))
        sample_ratings = ratings.sample(n=sample_size, random_state=42)

        actuals = sample_ratings['rating'].values
        user_ids = sample_ratings['user_id'].values
        item_ids = sample_ratings['item_id'].values

        logger.info(f"Predicting ratings for {sample_size} samples...")
        predictions = model.predict_batch(user_ids, item_ids)

        rmse = calculate_rmse(actuals, predictions)
        logger.info(f"ML Model (SVD) RMSE: {rmse:.4f}")

        # Compare with Baseline (1.3146)
        baseline_rmse = 1.3146
        if rmse < baseline_rmse:
            improvement = baseline_rmse - rmse
            logger.info(f"✅ SUCCESS: ML Model beat the baseline by {improvement:.4f}!")
        else:
            logger.info(f"❌ FAILED: ML Model did not beat the baseline. Current RMSE: {rmse:.4f}")

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise e

if __name__ == "__main__":
    main()
