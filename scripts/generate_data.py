import pandas as pd
import numpy as np
import random
from pathlib import Path

def generate_amazon_mimic_data():
    print("Generating professional synthetic dataset...")

    # Parameters
    n_users = 1000
    n_items = 500
    n_interactions = 10000
    categories = ['Electronics', 'Books', 'Fashion', 'Home & Kitchen', 'Beauty', 'Toys']
    brands = ['Sony', 'Samsung', 'Apple', 'Nike', 'Adidas', 'Logitech', 'generic']

    # 1. Generate Metadata
    items = []
    for i in range(n_items):
        cat = random.choice(categories)
        brand = random.choice(brands)
        price_map = {'Electronics': (50, 1000), 'Books': (10, 50), 'Fashion': (20, 200),
                    'Home & Kitchen': (15, 500), 'Beauty': (10, 100), 'Toys': (10, 100)}
        p_min, p_max = price_map[cat]

        items.append({
            'item_id': f'item_{i}',
            'title': f'Product {i} - {brand}',
            'category': cat,
            'price': round(random.uniform(p_min, p_max), 2),
            'brand': brand
        })

    df_metadata = pd.DataFrame(items)

    # 2. Generate Interactions
    interactions = []

    # GUARANTEE: Every item must be used at least once to avoid the "Total Items: 29" problem
    for i in range(n_items):
        user = f'user_{random.randint(0, n_users-1)}'
        interactions.append({
            'user_id': user,
            'item_id': f'item_{i}',
            'rating': random.randint(1, 5),
            'timestamp': 1600000000 + random.randint(0, 10000000)
        })

    # Fill the remaining interactions
    # We use a mix of uniform (discovery) and weighted (popularity)
    remaining = n_interactions - n_items

    # 70% Uniform distribution (ensures high item count)
    for _ in range(int(remaining * 0.7)):
        user = f'user_{random.randint(0, n_users-1)}'
        item_id = f'item_{random.randint(0, n_items-1)}'
        interactions.append({
            'user_id': user,
            'item_id': item_id,
            'rating': random.randint(1, 5),
            'timestamp': 1600000000 + random.randint(0, 10000000)
        })

    # 30% Popularity biased (creates the "Head" of the distribution)
    popular_items = [f'item_{i}' for i in range(50)] # Top 50 items
    for _ in range(remaining - int(remaining * 0.7)):
        user = f'user_{random.randint(0, n_users-1)}'
        item_id = random.choice(popular_items)
        interactions.append({
            'user_id': user,
            'item_id': item_id,
            'rating': random.randint(1, 5),
            'timestamp': 1600000000 + random.randint(0, 10000000)
        })

    df_ratings = pd.DataFrame(interactions)

    # Save files
    raw_dir = Path('data/raw')
    raw_dir.mkdir(parents=True, exist_ok=True)

    df_ratings.to_csv(raw_dir / 'ratings.csv', index=False)
    df_metadata.to_csv(raw_dir / 'metadata.csv', index=False)

    print(f"✅ Successfully generated {len(df_ratings)} interactions and {len(df_metadata)} items.")

if __name__ == '__main__':
    generate_amazon_mimic_data()
