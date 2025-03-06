import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
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
        self.feature_matrix, self.tfidf_vectorizer, self.province_encoder, self.category_encoder = self.create_feature_matrix()

    def load_data(self, file_path):
        """Load data from CSV file and preprocess tags"""
        data = pd.read_csv(file_path)
        data['Tags'] = data['Tags'].apply(self._process_tags)
        return data

    def _process_tags(self, tags):
        """Convert tags string to list and clean the tags"""
        if pd.isna(tags) or not isinstance(tags, str):
            return []
        
        def split_joined_tags(tag):
            split_words = ['natural', 'cultural', 'historical', 'recreational', 'tourist', 'site', 'area', 'spot', 'destination']
            for word in split_words:
                if word in tag:
                    parts = tag.split(word)
                    return [parts[0], word] if parts[0] else [word]
            return [tag]
        
        processed_tags = []
        for tag in tags.replace(',', ' ').split():
            processed_tags.extend(split_joined_tags(tag.strip().lower()))
        
        return list(dict.fromkeys(processed_tags))
    
    def _tags_to_string(self, tags):
        """Convert list of tags to a space-separated string"""
        return ' '.join(tags)

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

        :return: Feature matrix, TfidfVectorizer, province encoder, category encoder
        """
        self.data['Category'] = self.data['Tags'].apply(
            lambda x: [category for category, tags in self.categorized_tags.items() if any(tag in tags for tag in x)]
        )
        
        self.data['Category'] = self.data['Category'].apply(lambda x: ', '.join(x) if x else 'Uncategorized')
        
        province_encoder = LabelEncoder()
        category_encoder = LabelEncoder()
        self.data['Encoded_Province'] = province_encoder.fit_transform(self.data['Province'])
        self.data['Encoded_Category'] = category_encoder.fit_transform(self.data['Category'])

        self.data['Tags_str'] = self.data['Tags'].apply(self._tags_to_string)
        tfidf_vectorizer = TfidfVectorizer()
        tag_features = tfidf_vectorizer.fit_transform(self.data['Tags_str'])

        tag_features_df = pd.DataFrame(tag_features.toarray(),
                                       columns=tfidf_vectorizer.get_feature_names_out(),
                                       index=self.data.index)

        feature_matrix = pd.concat([
            self.data[['Encoded_Province', 'Encoded_Category']],
            tag_features_df
        ], axis=1)

        return feature_matrix, tfidf_vectorizer, province_encoder, category_encoder
    def get_available_tags(self, province=None, categories=None):
        """
        Get comprehensive list of tags for given province and categories.
        Searches for tags associated with each selected category individually,
        as well as tags from attractions matching any combination of the categories.
        
        :param province: Optional province to filter by
        :param categories: Optional single category or list of categories to filter by
        :return: Sorted list of unique tags
        """
        filtered_data = self.data.copy()
        all_tags = []
        
        # # Handle province filtering
        # if province:
        #     filtered_data = filtered_data[filtered_data['Province'].str.lower() == province.lower()]
        
        # If no categories provided, return all tags
        if not categories:
            all_tags = [tag for tags_list in filtered_data['Tags'] for tag in tags_list]
            return sorted(set(all_tags))
        
        # Convert categories to list if it's a string
        if not isinstance(categories, list):
            categories = [categories]
            print("converting to a list")
        # Convert categories to lowercase
        categories = [cat.lower() for cat in categories]
        print(categories)
        # 1. Find tags for each individual category
        for category in categories:
            # print(i)
            print("for", category)
            category_filtered = filtered_data[
                filtered_data['Category'].str.lower().str.contains(category)
            ]
            category_tags = [tag for tags_list in category_filtered['Tags'] for tag in tags_list]
            print("DEBUG:", len(category_tags))
            all_tags.extend(category_tags)
        
        # 2. Find tags for attractions matching ANY of the categories (combined search)
        if len(categories) > 1:
            combined_filtered = filtered_data[
                filtered_data['Category'].apply(
                    lambda x: any(cat in x.lower() for cat in categories)
                )
            ]
            combined_tags = [tag for tags_list in combined_filtered['Tags'] for tag in tags_list]
            all_tags.extend(combined_tags)
        
        # Return unique sorted tags
        return sorted(set(all_tags))
        
    def recommend(self, province=None, categories=None, tags=None, top_n=5):
        """
        Generate comprehensive recommendations based on user preferences
        with multi-category support and improved tag matching.
        
        :param province: Optional province to filter by
        :param categories: Optional single category or list of categories (can be comma-separated string)
        :param tags: Optional tags to filter by (can be comma-separated string)
        :param top_n: Number of top recommendations to return
        :return: DataFrame of recommendations
        """
        try:
            filtered_data = self.data.copy()
            
            # Normalize province
            province = province.lower() if province else None
            
            # Process categories - handle comma-separated string
            if categories:
                if isinstance(categories, str):
                    # Split by comma and strip whitespace
                    categories = [cat.strip() for cat in categories.split(',') if cat.strip()]
                # Ensure list and lowercase
                categories = [cat.lower() for cat in categories]
            
            # Process tags - handle comma-separated string
            if tags:
                if isinstance(tags, str):
                    # Split by comma and strip whitespace
                    tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
                # Ensure list and lowercase
                tags = [t.lower() for t in tags]

            # Filter by province
            if province:
                filtered_data = filtered_data[filtered_data['Province'].str.lower() == province]
            
            # Filter by categories - match any of the categories
            if categories:
                filtered_data = filtered_data[
                    filtered_data['Category'].apply(
                        lambda x: any(cat in x.lower() for cat in categories)
                    )
                ]
            
            # Filter and score by tags with improved matching
            if tags:
                def calculate_tag_match(attraction_tags, user_tags):
                    # Calculate match score with improved partial matching
                    matches = 0
                    for tag in user_tags:
                        # Check for exact match
                        if tag in attraction_tags:
                            matches += 1
                        else:
                            # Check for partial matches
                            for at in attraction_tags:
                                # Check if user tag is in attraction tag or vice versa
                                if tag in at or at in tag:
                                    matches += 0.8  # Partial match has slightly lower weight
                                    break
                    
                    return matches / len(user_tags) if user_tags else 0

                # Calculate tag match scores for all attractions
                filtered_data['tag_match_score'] = filtered_data['Tags'].apply(
                    lambda attraction_tags: calculate_tag_match(attraction_tags, tags)
                )
                
                # Filter to attractions with positive match scores and sort
                filtered_data = filtered_data[filtered_data['tag_match_score'] > 0].sort_values(
                    'tag_match_score', ascending=False
                )

            # Handle empty results
            if filtered_data.empty:
                print(f"No recommendations found for province={province}, categories={categories}, tags={tags}")
                return pd.DataFrame()

            # Prepare and return results
            result = filtered_data.head(top_n)[['ID', 'Name', 'Province', 'Category', 'Tags']]
            result['Tags'] = result['Tags'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
            return result

        except Exception as e:
            import traceback
            print(f"Error in recommend method: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()
            
if __name__ == "__main__":
    print("ContentBasedRecommender is ready to use.")
