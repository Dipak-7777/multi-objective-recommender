import pandas as pd
import logging
from pathlib import Path
from src.features.engineering import engineer_item_features, engineer_user_features

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def main():
    # ABSOLUTE PATHS: Ensuring zero environment errors
    base_path = Path(r'C:\Users\DrX DIPAK\projects\multi-objective-recommender')
    processed_path = base_path / 'data' / 'processed'

    try:
        logger.info("Loading analytical summaries...")
        user_summary = pd.read_csv(processed_path / 'user_summary.csv')
        item_summary = pd.read_csv(processed_path / 'item_summary.csv')

        # Step 1: Engineer Features
        logger.info("Starting Feature Engineering pipeline...")
        user_features = engineer_user_features(user_summary)
        item_features = engineer_item_features(item_summary)

        # Step 2: Save to Feature Store
        user_features.to_csv(processed_path / 'user_features.csv', index=False)
        item_features.to_csv(processed_path / 'item_features.csv', index=False)

        logger.info("✅ SUCCESS: Feature Store created!")
        logger.info(f"User Features: {processed_path / 'user_features.csv'}")
        logger.info(f"Item Features: {processed_path / 'item_features.csv'}")

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        raise e

if __name__ == "__main__":
    main()
