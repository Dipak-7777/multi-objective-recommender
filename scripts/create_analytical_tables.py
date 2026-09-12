import pandas as pd
import yaml
import logging
from pathlib import Path
from src.analytics.descriptive import create_user_summary, create_item_summary

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    processed_dir = Path(config['paths']['processed_data_dir'])

    logger.info("Starting Analytical Table Generation...")

    # 1. Load Processed Data
    try:
        ratings = pd.read_csv(processed_dir / 'ratings_processed.csv')
        metadata = pd.read_csv(processed_dir / 'metadata_processed.csv')
        logger.info("Processed data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load processed data: {e}")
        return

    # 2. Create Summaries
    try:
        user_summary = create_user_summary(ratings, metadata)
        item_summary = create_item_summary(ratings, metadata)

        # 3. Save Analytical Tables
        user_summary.to_csv(processed_dir / 'user_summary.csv', index=False)
        item_summary.to_csv(processed_dir / 'item_summary.csv', index=False)

        logger.info(f"✅ Success! Analytical tables saved to {processed_dir}")
        logger.info(f"User Summary shape: {user_summary.shape}")
        logger.info(f"Item Summary shape: {item_summary.shape}")

    except Exception as e:
        logger.error(f"Error during aggregation: {e}")

if __name__ == '__main__':
    main()
