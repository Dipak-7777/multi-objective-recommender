import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def validate_ratings(df: pd.DataFrame):
    """
    Validates the ratings dataframe.
    Returns a tuple: (is_valid, error_message)
    """
    # 1. Check for critical nulls
    critical_cols = ['user_id', 'item_id', 'rating']
    for col in critical_cols:
        if df[col].isnull().any():
            return False, f"Critical column {col} contains null values."

    # 2. Check rating range (1-5)
    if not df['rating'].between(1, 5).all():
        invalid_count = (~df['rating'].between(1, 5)).sum()
        return False, f"Found {invalid_count} ratings outside the 1-5 range."

    return True, "Ratings validation passed."

def validate_metadata(df: pd.DataFrame):
    """
    Validates the metadata dataframe.
    Returns a tuple: (is_valid, error_message)
    """
    # 1. Check for critical nulls
    if df['item_id'].isnull().any():
        return False, "Critical column item_id contains null values."

    # 2. Check for invalid prices
    if (df['price'] <= 0).any():
        invalid_count = (df['price'] <= 0).sum()
        return False, f"Found {invalid_count} items with price <= 0."

    return True, "Metadata validation passed."
