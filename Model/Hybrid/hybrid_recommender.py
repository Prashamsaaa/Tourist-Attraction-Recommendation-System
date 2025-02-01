import logging
import pandas as pd

class HybridRecommender:
    def __init__(self, content_recommender, distilbert_recommender, ncf_recommender, distilbert_weight=0.6):
        """
        Initialize the Hybrid Recommender.

        :param content_recommender: Instance of ContentBasedRecommender.
        :param distilbert_recommender: Instance of DistilBERTRecommender.
        :param ncf_recommender: Instance of NCF Recommender.
        :param distilbert_weight: Weight for DistilBERT recommendations in final score calculation.
        """
        self.content_recommender = content_recommender
        self.distilbert_recommender = distilbert_recommender
        self.ncf_recommender = ncf_recommender
        self.distilbert_weight = distilbert_weight
        self.ncf_weight = 1 - distilbert_weight

    def recommend_for_new_user(self, province=None, category=None, tags=None):
        """Generate recommendations for new users using content-based filtering."""
        try:
            # Normalize inputs
            province = province.strip() if province else None
            category = category.strip() if category else None
            
            # Process tags
            if isinstance(tags, str):
                tags = [tags]
            elif not tags:
                tags = []
            
            # Get recommendations
            recommendations = self.content_recommender.recommend(
                province=province,
                category=category,
                tags=tags,
                top_n=5
            )
            
            if recommendations.empty:
                logging.info(f"No recommendations found for province={province}, category={category}, tags={tags}")
                return pd.DataFrame()
            
            return recommendations

        except Exception as e:
            logging.error(f"Error generating recommendations for new user: {e}")
            return pd.DataFrame()

    def recommend_for_old_user(self, user_id, embeddings, descriptions, ratings, preferred_province):
        """
        Generate hybrid recommendations for old users by combining NCF and DistilBERT scores.

        :param user_id: ID of the user for whom recommendations are generated.
        :param embeddings: Precomputed embeddings for all descriptions.
        :param descriptions: DataFrame containing place descriptions.
        :param ratings: DataFrame containing user-item interaction data.
        :param preferred_province: User's preferred province to filter recommendations.
        :return: DataFrame containing hybrid recommendations for old users.
        """
        try:
            # First filter descriptions by province
            filtered_descriptions = descriptions[
                descriptions['Province'].str.lower() == preferred_province.lower()
            ].copy()

            if filtered_descriptions.empty:
                print(f"No attractions found in province: {preferred_province}")
                return pd.DataFrame()

            # Collect valid item IDs for the chosen province
            valid_item_ids = set(filtered_descriptions['ID'].astype(int).tolist())

            # Get NCF recommendations, passing valid_item_ids
            ncf_recs = self.ncf_recommender.recommend(
                user_id=user_id,
                top_n=20,  # get more items, then merge with DistilBERT
                valid_item_ids=valid_item_ids
            )
            ncf_df = pd.DataFrame(ncf_recs, columns=['item_id', 'score'])
            
            # Filter NCF recommendations by province
            ncf_filtered = ncf_df[ncf_df['item_id'].isin(filtered_descriptions['ID'])]

            # Get DistilBERT recommendations
            distilbert_recs = self.distilbert_recommender.recommend_places(
                user_id=user_id,
                data=filtered_descriptions,  # Pass filtered descriptions
                ratings=ratings,
                embeddings=embeddings,
                top_n=10
            )

            if distilbert_recs.empty and ncf_filtered.empty:
                print(f"No recommendations found for user {user_id} in province {preferred_province}")
                return pd.DataFrame()

            # Merge recommendations
            combined_df = distilbert_recs.merge(
                ncf_filtered,
                left_on='ID',
                right_on='item_id',
                how='outer'
            )

            # Normalize and calculate final scores only for places in the preferred province
            combined_df = combined_df[combined_df['Province'].str.lower() == preferred_province.lower()]

            if combined_df.empty:
                return pd.DataFrame()

            # Normalize scores between 0 and 1
            for col in ['score', 'DistilBERT_Score']:
                if col in combined_df.columns:
                    score_range = combined_df[col].max() - combined_df[col].min()
                    if score_range > 0:
                        combined_df[f'{col}_norm'] = (combined_df[col] - combined_df[col].min()) / score_range
                    else:
                        combined_df[f'{col}_norm'] = 0

            # Calculate final score
            combined_df['Final_Score'] = (
                self.distilbert_weight * combined_df.get('DistilBERT_Score_norm', 0).fillna(0) +
                self.ncf_weight * combined_df.get('score_norm', 0).fillna(0)
            )

            # Return top 5 recommendations with all relevant information
            result = combined_df.nlargest(5, 'Final_Score')[
                ['ID', 'Name', 'Province', 'Category', 'Final_Score']
            ]
            return result

        except Exception as e:
            logging.error(f"Error generating hybrid recommendations for user {user_id}: {e}")
            return pd.DataFrame()

