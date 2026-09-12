import pandas as pd
import yaml
import logging
from pathlib import Path
from src.data.validation import validate_ratings, validate_metadata
from src.data.preprocessing import clean_ratings, clean_metadata

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    raw_dir = Path(config['paths']['raw_data_dir'])
    processed_dir = Path(config['paths']['processed_data_dir'])
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Data Preprocessing Pipeline...")

    # 1. Load Raw Data
    try:
        ratings_raw = pd.read_csv(raw_dir / 'ratings.csv', sep=None, engine='python', on_bad_lines='skip')
        metadata_raw = pd.read_csv(raw_dir / 'metadata.csv', sep=None, engine='python', on_bad_lines='skip')
        logger.info("Raw data loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")
        return

    # 2. Cleaning Phase
    logger.info("Cleaning data...")
    ratings_cleaned = clean_ratings(ratings_raw)
    metadata_cleaned = clean_metadata(metadata_raw)

    # 3. Validation Phase (The Guardrail)
    logger.info("Validating cleaned data...")

    r_valid, r_msg = validate_ratings(ratings_cleaned)
    m_valid, m_msg = validate_metadata(metadata_cleaned)

    if not r_valid:
        logger.error(f"Ratings Validation Failed: {r_msg}")
        return
    if not m_valid:
        logger.error(f"Metadata Validation Failed: {m_msg}")
        return

    logger.info("All validations passed.")

    # 4. Save Processed Data
    ratings_cleaned.to_csv(processed_dir / 'ratings_processed.csv', index=False)
    metadata_cleaned.to_csv(processed_dir / 'metadata_processed.csv', index=False)

    logger.info(f"✅ Success! Processed data saved to {processed_dir}")

if __name__ == '__main__':
    main()
