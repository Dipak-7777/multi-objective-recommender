import pandas as pd
import logging

logger = logging.getLogger(__name__)

def clean_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs cleaning on the ratings dataframe.
    """
    initial_len = len(df)

    # Remove duplicates
    df = df.drop_duplicates()

    # Ensure correct types
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # Drop rows where cast failed (NaNs)
    df = df.dropna()

    final_len = len(df)
    logger.info(f"Cleaned ratings: Removed {initial_len - final_len} rows.")
    return df

def clean_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs cleaning on the metadata dataframe.
    """
    initial_len = len(df)

    # Remove duplicates based on item_id
    df = df.drop_duplicates(subset=['item_id'])

    # Handle missing values for non-critical columns
    # For title, category, brand: fill with 'Unknown'
    cols_to_fix = ['title', 'category', 'brand']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # Ensure price is numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price']) # Drop if price is totally missing

    final_len = len(df)
    logger.info(f"Cleaned metadata: Removed {initial_len - final_len} rows.")
    return df
