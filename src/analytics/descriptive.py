import pandas as pd
import logging

logger = logging.getLogger(__name__)

def create_user_summary(ratings: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary table of user behavior.
    """
    logger.info("Generating user summary table...")

    # Merge ratings with metadata to get categories
    df = ratings.merge(metadata[['item_id', 'category']], on='item_id', how='left')

    user_summary = df.groupby('user_id').agg(
        total_interactions=('rating', 'count'),
        avg_rating=('rating', 'mean'),
        unique_categories=('category', 'nunique')
    ).reset_index()

    # Calculate engagement rate (relative to the most active user)
    max_interactions = user_summary['total_interactions'].max()
    user_summary['engagement_score'] = user_summary['total_interactions'] / max_interactions

    return user_summary

def create_item_summary(ratings: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a summary table of item performance.
    """
    logger.info("Generating item summary table...")

    item_summary = ratings.groupby('item_id').agg(
        total_interactions=('rating', 'count'),
        avg_rating=('rating', 'mean'),
        unique_users=('user_id', 'nunique')
    ).reset_index()

    # Merge with metadata for business attributes
    item_summary = item_summary.merge(metadata[['item_id', 'category', 'price', 'brand']], on='item_id', how='left')

    return item_summary
