import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MultiObjectiveReranker:
    """
    Re-ranks a list of recommendations based on multiple objectives:
    1. Relevance (Predicted Rating)
    2. Novelty (Inverse Popularity)
    3. Diversity (Category Spread)
    """
    def __init__(self, item_summary: pd.DataFrame, w_relevance=0.4, w_novelty=0.4, w_diversity=0.2):
        self.item_summary = item_summary
        self.w_relevance = w_relevance
        self.w_novelty = w_novelty
        self.w_diversity = w_diversity

        # Pre-calculate normalized popularity for Novelty score
        # Novelty = 1 - (item_interactions / max_interactions)
        max_pop = item_summary['total_interactions'].max()
        self.item_novelty = 1 - (item_summary.set_index('item_id')['total_interactions'] / max_pop)

    def rerank(self, user_id, initial_recs: list, n=10):
        """
        Re-ranks initial_recs (list of (item_id, score)) based on multiple objectives.
        """
        if not initial_recs:
            return []

        # Convert initial recs to a DataFrame for easier manipulation
        df = pd.DataFrame(initial_recs, columns=['item_id', 'relevance_score'])

        # Merge with item metadata for diversity and novelty
        df = df.merge(self.item_summary[['item_id', 'category']], on='item_id', how='left')
        df['novelty_score'] = df['item_id'].map(self.item_novelty).fillna(0)

        # Normalize relevance score to 0-1 range for weighted sum
        min_rel = df['relevance_score'].min()
        max_rel = df['relevance_score'].max()
        if max_rel != min_rel:
            df['norm_relevance'] = (df['relevance_score'] - min_rel) / (max_rel - min_rel)
        else:
            df['norm_relevance'] = 1.0

        # Reranking loop (Greedy selection for diversity)
        reranked_list = []
        remaining_items = df.to_dict('records')
        seen_categories = set()

        while len(reranked_list) < n and remaining_items:
            best_score = -float('inf')
            best_item_idx = -1

            for idx, item in enumerate(remaining_items):
                # 1. Relevance Component
                rel = item['norm_relevance']

                # 2. Novelty Component
                nov = item['novelty_score']

                # 3. Diversity Component
                # If category is already seen, apply a penalty
                div = 1.0 if item['category'] not in seen_categories else 0.0

                # Weighted Multi-Objective Score
                final_score = (self.w_relevance * rel) + (self.w_novelty * nov) + (self.w_diversity * div)

                if final_score > best_score:
                    best_score = final_score
                    best_item_idx = idx

            # Add the best item to the list
            chosen_item = remaining_items.pop(best_item_idx)
            reranked_list.append(chosen_item['item_id'])
            seen_categories.add(chosen_item['category'])

        return reranked_list
