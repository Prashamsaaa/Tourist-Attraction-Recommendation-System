import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import json

class ContentBasedRecommender:
    def __init__(self, data_path, tags_path):
        """
        Initialize the Content-Based Recommender
        
        :param data_path: Path to the CSV file containing attraction data
        :param tags_path: Path to the JSON file containing categorized tags
        """
        self.data = self.load_data(data_path)
        self.categorized_tags = self.load_categorized_tags(tags_path)
        self.feature_matrix, self.mlb, self.province_encoder, self.category_encoder = self.create_feature_matrix()

    def load_data(self, file_path):
        """Load data from CSV file and preprocess tags"""
        data = pd.read_csv(file_path)
        # Ensure tags are properly processed into lists
        data['Tags'] = data['Tags'].apply(self._process_tags)
        return data

    def _process_tags(self, tags):
        """Convert tags string to list and clean the tags"""
        if pd.isna(tags) or not isinstance(tags, str):
            return []
        # Split by space or comma and clean each tag
        return [tag.strip().lower() for tag in tags.replace(',', ' ').split() if tag.strip()]

    def load_categorized_tags(self, json_path):
        """
        Load categorized tags from JSON file
        
        :param json_path: Path to the JSON file
        :return: Dictionary of categorized tags
        """
        with open(json_path, 'r') as f:
            return json.load(f)

    def create_feature_matrix(self):
        """
        Create feature matrix for attractions
        
        :return: Feature matrix, MultiLabelBinarizer, province encoder, category encoder
        """
        # Assign categories to attractions based on their tags
        self.data['Category'] = self.data['Tags'].apply(
            lambda x: next((category for category, tags in self.categorized_tags.items() 
                            if any(tag in tags for tag in x)), 'Uncategorized')
        )
        
        # Encode provinces and categories
        province_encoder = LabelEncoder()
        category_encoder = LabelEncoder()
        self.data['Encoded_Province'] = province_encoder.fit_transform(self.data['Province'])
        self.data['Encoded_Category'] = category_encoder.fit_transform(self.data['Category'])
        
        # Encode tags using MultiLabelBinarizer
        mlb = MultiLabelBinarizer()
        tag_features = pd.DataFrame(mlb.fit_transform(self.data['Tags']), 
                                    columns=mlb.classes_, 
                                    index=self.data.index)
        
        # Combine all features into a single feature matrix
        feature_matrix = pd.concat([
            self.data[['Encoded_Province', 'Encoded_Category']], 
            tag_features
        ], axis=1)
        
        return feature_matrix, mlb, province_encoder, category_encoder

    def recommend(self, province, category, tags, top_n=5):
        """Generate recommendations based on user preferences"""
        try:
            # Start with a copy of the data
            filtered_data = self.data.copy()

            # Normalize the tags in the dataset
            filtered_data['Tags'] = filtered_data['Tags'].apply(
                lambda x: [t.lower().strip() for t in (x if isinstance(x, list) else [])]
            )

            # Filter by province
            if province:
                filtered_data = filtered_data[
                    filtered_data['Province'].str.lower() == province.lower()
                ]

            # Filter by category
            if category:
                filtered_data = filtered_data[
                    filtered_data['Category'].str.lower() == category.lower()
                ]

            # Process tags
            if tags:
                # Normalize input tags
                if isinstance(tags, str):
                    user_tags = {tags.lower().strip()}
                else:
                    user_tags = {t.lower().strip() for t in tags}

                # Calculate tag matching score
                filtered_data['tag_match_score'] = filtered_data['Tags'].apply(
                    lambda x: sum(1 for tag in x if tag in user_tags) / len(user_tags)
                )

                # Keep only places with at least one matching tag
                filtered_data = filtered_data[filtered_data['tag_match_score'] > 0]
                filtered_data = filtered_data.sort_values('tag_match_score', ascending=False)

            if filtered_data.empty:
                print(f"No matches found for province={province}, category={category}, tags={tags}")
                return pd.DataFrame()

            # Select and format results
            result = filtered_data.head(top_n)[['ID', 'Name', 'Province', 'Category', 'Tags']]
            
            # Format tags for display
            result['Tags'] = result['Tags'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
            
            return result

        except Exception as e:
            print(f"Error in recommend method: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    print("ContentBasedRecommender is ready to use.")
