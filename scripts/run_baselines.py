import pandas as pd
import numpy as np
import logging
from pathlib import Path
from src.recommenders.baselines import PopularityBaseline, UserAverageBaseline, calculate_rmse

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

        # --- BASELINE 1: Popularity ---
        pop_model = PopularityBaseline()
        pop_model.fit(ratings)

        # For a popularity baseline, we typically measure 'Recall' or 'Precision'.
        # But for a simple numerical baseline, we'll see the top 5 items.
        top_5 = pop_model.predict(user_id=None, n=5)
        logger.info(f"Top 5 Popular Items (Baseline): {top_5}")

        # --- BASELINE 2: User Average ---
        avg_model = UserAverageBaseline()
        avg_model.fit(ratings)

        # Evaluate User Average on the existing dataset
        actuals = []
        predictions = []

        # To keep it fast, we sample 1000 interactions for evaluation
        sample_ratings = ratings.sample(n=min(1000, len(ratings)), random_state=42)

        for _, row in sample_ratings.iterrows():
            actuals.append(row['rating'])
            predictions.append(avg_model.predict(row['user_id']))

        rmse = calculate_rmse(actuals, predictions)
        logger.info(f"User-Average Baseline RMSE: {rmse:.4f}")
        logger.info("Note: This RMSE is the 'Low Bar'. Our ML models MUST beat this number.")

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise e

if __name__ == "__main__":
    main()
