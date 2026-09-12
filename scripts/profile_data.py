import pandas as pd
import yaml
import numpy as np
from pathlib import Path

def load_config():
    with open('configs/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    raw_dir = Path(config['paths']['raw_data_dir'])

    try:
        # Removed low_memory=False because it is incompatible with engine='python'
        # sep=None + engine='python' handles auto-detection of the delimiter
        ratings = pd.read_csv(raw_dir / 'ratings.csv', sep=None, engine='python', on_bad_lines='skip')
        metadata = pd.read_csv(raw_dir / 'metadata.csv', sep=None, engine='python', on_bad_lines='skip')
    except Exception as e:
        print(f"Critical Error loading CSVs: {e}")
        return

    # Basic check to ensure columns exist
    if 'user_id' not in ratings.columns or 'item_id' not in ratings.columns:
        print(f"Error: Expected columns not found in ratings.csv. Found: {ratings.columns.tolist()}")
        return

    n_users = ratings['user_id'].nunique()
    n_items = ratings['item_id'].nunique()
    n_interactions = len(ratings)

    possible_interactions = n_users * n_items
    sparsity = 1 - (n_interactions / possible_interactions) if possible_interactions > 0 else 1

    user_counts = ratings.groupby('user_id').size()
    item_counts = ratings.groupby('item_id').size()
    missing_meta = metadata.isnull().sum().to_dict()

    report = f"""# Data Profiling Report

## 1. General Dimensions
- **Total Users:** {n_users}
- **Total Items:** {n_items}
- **Total Interactions:** {n_interactions}
- **Matrix Sparsity:** {sparsity:.2%}

## 2. User Activity
- **Avg interactions per user:** {n_interactions/n_users:.2f}
- **Max interactions by a single user:** {user_counts.max()}
- **Min interactions by a user:** {user_counts.min()}
- **Users with only 1 interaction:** {(user_counts == 1).sum()} ({((user_counts == 1).sum()/n_users):.2%})

## 3. Item Popularity
- **Avg interactions per item:** {n_interactions/n_items:.2f}
- **Max interactions for a single item:** {item_counts.max()}
- **Items with only 1 interaction (Cold Start):** {(item_counts == 1).sum()} ({((item_counts == 1).sum()/n_items):.2%})

## 4. Metadata Quality
- **Missing Values per Column:**
{missing_meta}

## 5. Rating Distribution
{ratings['rating'].value_counts(normalize=True).to_string()}
"""

    report_path = Path('reports/data_profiling.md')
    report_path.write_text(report)
    print(f"\n✅ Profiling complete. Report saved to {report_path}")

if __name__ == '__main__':
    main()
