import pandas as pd
import sqlite3
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def migrate_csv_to_sqlite():
    """
    Migrates processed CSV files into a SQLite database.
    This is the first step in turning a data science project into a software product.
    """
    base_path = Path(r'C:\Users\DrX DIPAK\projects\multi-objective-recommender')
    processed_path = base_path / 'data' / 'processed'
    db_path = base_path / 'data' / 'recommender.db'

    # Files to migrate
    files_to_migrate = {
        'ratings_processed.csv': 'ratings',
        'metadata_processed.csv': 'metadata',
        'user_summary.csv': 'user_summary',
        'item_summary.csv': 'item_summary',
        'user_features.csv': 'user_features',
        'item_features.csv': 'item_features'
    }

    try:
        # Connect to SQLite database
        conn = sqlite3.connect(db_path)
        logger.info(f"Connected to database at {db_path}")

        for csv_file, table_name in files_to_migrate.items():
            csv_path = processed_path / csv_file
            if csv_path.exists():
                logger.info(f"Migrating {csv_file} to table {table_name}...")
                df = pd.read_csv(csv_path)
                df.to_sql(table_name, conn, if_exists='replace', index=False)
            else:
                logger.warning(f"File {csv_file} not found, skipping.")

        conn.close()
        logger.info("✅ DATABASE MIGRATION COMPLETE!")
        logger.info(f"All data now stored in: {db_path}")

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise e

if __name__ == "__main__":
    migrate_csv_to_sqlite()
