import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def engineer_item_features(item_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms item summaries into ML-ready features.
    """
    logger.info("Engineering item features...")
    df = item_summary.copy()

    # 1. Category Encoding (Text -> Number)
    le_cat = LabelEncoder()
    df['category_encoded'] = le_cat.fit_transform(df['category'].astype(str))

    # 2. Brand Encoding (Text -> Number)
    le_brand = LabelEncoder()
    df['brand_encoded'] = le_brand.fit_transform(df['brand'].astype(str))

    # 3. Price Normalization (0 to 1 scale)
    scaler = MinMaxScaler()
    df['price_scaled'] = scaler.fit_transform(df[['price']].fillna(df['price'].median()))

    # 4. Popularity Normalization
    df['popularity_score'] = scaler.fit_transform(df[['total_interactions']])

    # Drop raw text columns to keep only numerical features
    features = df[['item_id', 'category_encoded', 'brand_encoded', 'price_scaled', 'popularity_score', 'avg_rating']]

    return features

def engineer_user_features(user_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms user summaries into ML-ready features.
    """
    logger.info("Engineering user features...")
    df = user_summary.copy()

    scaler = MinMaxScaler()

    # Normalize engagement and diversity
    df['engagement_scaled'] = scaler.fit_transform(df[['total_interactions']])
    df['diversity_scaled'] = scaler.fit_transform(df[['unique_categories']])

    # Fill NaNs in avg_rating with the global median
    df['avg_rating'] = df['avg_rating'].fillna(df['avg_rating'].median())

    features = df[['user_id', 'engagement_scaled', 'diversity_scaled', 'avg_rating']]

    return features
