import logging
import pandas as pd
import numpy as np
import torch
from NCF.recommendation import generate_recommendations
from sklearn.preprocessing import LabelEncoder
from NCF.dataset import load_and_preprocess_data
from sklearn.metrics.pairwise import cosine_similarity


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
        """
        Generate recommendations for new users using content-based filtering.
        """
        try:
            # Normalize inputs
            province = province.strip() if province else None
            category = category.strip() if category else None

            # Process tags
            if isinstance(tags, str):
                tags = [tags]
            elif not tags:
                tags = []

            # Get recommendations using content-based filtering
            recommendations = self.content_recommender.recommend(
                province=province,
                categories=category,
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

    # def recommend_for_old_user(self, user_id, descriptions, ratings, preferred_province, user_encoder, place_encoder):
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
            # Step 1: Filter descriptions by province
            filtered_descriptions = descriptions[
                    descriptions['Province'].str.lower() == preferred_province.lower()
                ].copy()

            if filtered_descriptions.empty:
                    print(f"No attractions found in province: {preferred_province}")
                    return pd.DataFrame()

                # print(filtered_descriptions.head())
                # Collect valid item IDs for the chosen province
            valid_item_ids = set(filtered_descriptions['ID'].astype(int).tolist())

                # Step 2: Get NCF recommendations filtered by valid item IDs
                    # Load and preprocess data
            # place_encoder = LabelEncoder()
            # attraction_df = pd.read_csv('./Data/FinalDataset/Data.csv')
            # place_encoder = LabelEncoder()

            # ratings['id'] = place_encoder.fit_transform(ratings['id'])


            # Before passing to model, verify indices are within range
            print(f"Embedding size: {self.ncf_recommender.gmf.item_embedding.num_embeddings}")

            encoded_user_id = user_encoder.transform([user_id])[0]

            ncf_recs = generate_recommendations(self.ncf_recommender, encoded_user_id, num_items= len(place_encoder.classes_),top_k =10, device = 'cpu', place_encoder=place_encoder, attraction_df= filtered_descriptions)
            print(ncf_recs)
                # ncf_recs = generate_recommendations(self.ncf_recommender, user_id, num_items= len(place_encoder.classes_), device = 'cpu', place_encoder=place_encoder, attraction_df= filtered_descriptions)
            
            ncf_df = pd.DataFrame(ncf_recs, columns=['id', 'confidence'])
            print("NCF RECOMMENDATIONS GENERATION DONE")
            print(ncf_df.head())
            # return ncf_df
            # Step 3: Get DistilBERT recommendations using filtered descriptions and embeddings
            embeddings = filtered_descriptions['embeddings']
            distilbert_recs = self.distilbert_recommender.recommend_places(
                user_id=user_id,
                data=filtered_descriptions,
                ratings=ratings,
                embeddings=embeddings,
                top_n=10
            )
            print("DISTIL BERT RECOMMEDATION DONEEE")
            if distilbert_recs.empty and ncf_df.empty:
                print(f"No recommendations found for user {user_id} in province {preferred_province}")
                return pd.DataFrame()

            # Step 4: Merge NCF and DistilBERT recommendations
            combined_df = distilbert_recs.merge(
                ncf_df,
                left_on='id',
                right_on='id',
                how='outer'
            )
            # print(combined_df.head())

            # Normalize scores between 0 and 1 (for both NCF and DistilBERT)
            for col in ['confidence', 'DistilBERT_Score']:
                if col in combined_df.columns:
                    score_range = combined_df[col].max() - combined_df[col].min()
                    if score_range > 0:
                        combined_df[f'{col}_norm'] = (combined_df[col] - combined_df[col].min()) / score_range
                    else:
                        combined_df[f'{col}_norm'] = 0
            print(combined_df.head())
            # Calculate final score as a weighted combination of NCF and DistilBERT scores
            combined_df['Final_Score'] = (
                self.distilbert_weight * combined_df.get('DistilBERT_Score_norm', 0).fillna(0) +
                self.ncf_weight * combined_df.get('confidence_norm', 0).fillna(0)
            )

            # Filter results by preferred province (if necessary)
            combined_df = combined_df[combined_df['Province'].str.lower() == preferred_province.lower()]
            print(combined_df.info())
            if combined_df.empty:
                return pd.DataFrame()

            # Return top 5 recommendations with relevant information
            result = combined_df.nlargest(5, 'Final_Score')[
                ['id', 'Name', 'Province', 'Tags', 'Final_Score']
            ]

            return result

        except Exception as e:
            logging.error(f"Error generating hybrid recommendations for user {user_id}: {e}")
            return pd.DataFrame()
        
        
    def recommend_for_old_user(self, user_id, descriptions_all, ratings, preferred_province, user_encoder, place_encoder):
        """
        Generate hybrid recommendations by first combining scores before selecting top-k
        """
            # 2. Filter using vectorized operations
        descriptions = descriptions_all[
        descriptions_all['ID'].astype(str).str.lower()
        .isin(ratings['id'].astype(str).str.lower())
    ]

        # Get valid original IDs across all provinces
        valid_original_ids = descriptions[
            descriptions['ID'].astype(int).isin(ratings['id'].astype(int))
        ]['ID'].tolist()
        print(len(valid_original_ids))
        # Get already rated items (all provinces)
        user_ratings = ratings[(ratings['user_id'] == user_id)]
        rated_ids = user_ratings['id'].tolist()
        
        encoded_user_id = user_encoder.transform([user_id])[0]
        num_items = len(place_encoder.classes_)
        print(num_items)
        # Get NCF scores for all items
        try:
            with torch.no_grad():
                user_vector = torch.tensor([user_id], dtype=torch.long).repeat(num_items)
                all_item_ids = torch.arange(num_items, dtype=torch.long)
                ncf_scores = self.ncf_recommender(user_vector, all_item_ids).squeeze()
        except Exception as e:
            print(f"NCF failed: {str(e)}")
            ncf_scores = np.zeros(num_items)

        # Get DistilBERT scores for all items
        embedding_map = dict(zip(descriptions['ID'], descriptions['embeddings']))
        distilbert_scores = np.zeros(len(valid_original_ids))

        if not user_ratings.empty:
            try:
                for _, row in user_ratings.iterrows():
                    place_id = row['id']
                    if place_id not in embedding_map:
                        continue
                    
                    place_embedding = np.array(embedding_map[place_id]).reshape(1, -1)
                    embedding_subset = [np.array(embedding_map[item_id]) for item_id in valid_original_ids]
                    embedding_subset = np.vstack(embedding_subset)
                    
                    similarities = cosine_similarity(place_embedding, embedding_subset).flatten()
                    scores = 0.7 * similarities + 0.3 * row['rating']
                    distilbert_scores = np.maximum(distilbert_scores, scores)
                    
            except Exception as e:
                print(f"DistilBERT scoring failed: {str(e)}")
        print("DISTIL BERT DONE")
        print(len(ncf_scores))

        #    Create scoring dataframe with ID and scores
        scores = pd.DataFrame({
    'id': valid_original_ids,  # IDs that exist in both datasets
    'ncf_score': ncf_scores[:len(valid_original_ids)],
    'distilbert_score': distilbert_scores[:len(valid_original_ids)]
        })

    # Merge with FULL descriptions dataframe using inner join
        score_df = scores.merge(
    descriptions,
    left_on='id',
    right_on='ID',  # Match descriptions' ID column
    how='inner'
    ).drop(columns=['ID']) 

        # Apply province filtering after scoring
        province_lower = preferred_province.lower()
        score_df = score_df[score_df['Province'].str.lower() == province_lower]
        
        if score_df.empty:
            print(f"No attractions in {preferred_province}")
            return pd.DataFrame()

        # Exclude rated items
        score_df = score_df[~score_df['id'].isin(rated_ids)]

        # Normalize scores
        for col in ['ncf_score', 'distilbert_score']:
            col_min = score_df[col].min()
            col_max = score_df[col].max()
            if col_max - col_min > 0:
                score_df[f'{col}_norm'] = (score_df[col] - col_min) / (col_max - col_min)
            else:
                score_df[f'{col}_norm'] = 0.5

        # Calculate weighted scores
        score_df['final_score'] = (
            self.ncf_weight * score_df['ncf_score_norm'] +
            self.distilbert_weight * score_df['distilbert_score_norm']
        )
        # Get top 5 entries by final_score
        top_5_scores = score_df.nlargest(5, 'final_score').copy()

        result = top_5_scores[['id', 'Name', 'Province', 'Tags','Description', 'final_score']]

        return result.reset_index(drop=True)
