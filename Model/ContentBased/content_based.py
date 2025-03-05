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
        # Ensure tags are properly processed into lists
        data['Tags'] = data['Tags'].apply(self._process_tags)
        return data

    def _process_tags(self, tags):
        """Convert tags string to list and clean the tags"""
        if pd.isna(tags) or not isinstance(tags, str):
            return []
        
        # Custom tag splitting to handle joined words
        def split_joined_tags(tag):
            # List of common tag prefixes/suffixes to split
            split_words = ['natural', 'cultural', 'historical', 'recreational', 'tourist', 'site', 'area', 'spot', 'destination']
            
            # Try to split the tag
            for word in split_words:
                if word in tag:
                    parts = tag.split(word)
                    return [parts[0], word] if parts[0] else [word]
            
            # If no split found, return the original tag
            return [tag]
        
        # Split by comma or space, then further process each tag
        processed_tags = []
        for tag in tags.replace(',', ' ').split():
            processed_tags.extend(split_joined_tags(tag.strip().lower()))
        
        # Remove duplicates while preserving order
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

        # Encode tags using TfidfVectorizer
        self.data['Tags_str'] = self.data['Tags'].apply(self._tags_to_string)
        tfidf_vectorizer = TfidfVectorizer()
        tag_features = tfidf_vectorizer.fit_transform(self.data['Tags_str'])

        # Convert sparse matrix to DataFrame
        tag_features_df = pd.DataFrame(tag_features.toarray(),
                                       columns=tfidf_vectorizer.get_feature_names_out(),
                                       index=self.data.index)

        # Combine all features into a single feature matrix
        feature_matrix = pd.concat([ 
            self.data[['Encoded_Province', 'Encoded_Category']],
            tag_features_df
        ], axis=1)

        return feature_matrix, tfidf_vectorizer, province_encoder, category_encoder
    
    def get_available_tags(self, province=None, category=None):
        """
        Get available tags for a given province and category combination
        
        :param province: Province to filter by (optional)
        :param category: Category to filter by (optional)
        :return: List of unique tags
        """
        # Start with full dataset
        filtered_data = self.data.copy()
        
        # Filter by province if provided
        if province:
            filtered_data = filtered_data[
                filtered_data['Province'].str.lower() == province.lower()
            ]
        
        # Filter by category if provided
        if category:
            filtered_data = filtered_data[
                filtered_data['Category'].str.lower() == category.lower()
            ]
        
        # Flatten and get unique tags
        all_tags = [tag for tags_list in filtered_data['Tags'] for tag in tags_list]
        return sorted(set(all_tags))

    def recommend(self, province=None, category=None, tags=None, top_n=5):
        """Generate recommendations based on user preferences"""
        try:
            # Start with a copy of the data
            filtered_data = self.data.copy()

            # Normalize the tags in the dataset
            filtered_data['Tags'] = filtered_data['Tags'].apply(
                lambda x: [t.lower().strip() for t in (x if isinstance(x, list) else [])]
            )

            # Normalize and process input parameters
            province = province.lower() if province else None
            category = category.lower() if category else None
            
            # Process input tags
            if tags:
                # Convert tags to list if string
                if isinstance(tags, str):
                    tags = [tags.lower().strip()]
                else:
                    tags = [t.lower().strip() for t in tags]

            # Filter by province if provided
            if province:
                filtered_data = filtered_data[
                    filtered_data['Province'].str.lower() == province
                ]

            # Filter by category if provided
            if category:
                filtered_data = filtered_data[
                    filtered_data['Category'].str.lower() == category
                ]

            # Filter by tags if provided
            if tags:
                # Detailed tag matching
                def calculate_tag_match(attraction_tags, user_tags):
                    matches = []
                    for tag in user_tags:
                        # Comprehensive tag matching
                        tag_matches = [
                            1 for at in attraction_tags 
                            if (tag == at or  # Exact match
                                tag in at or  # Substring match
                                at in tag or  # Reverse substring match
                                any(tag == part for part in at.split('-')) or  # Hyphenated tag parts
                                any(part in tag for part in at.split('-'))  # Hyphenated tag parts
                            )
                        ]
                        matches.extend(tag_matches)
                    
                    # Normalize match score
                    return len(matches) / len(user_tags) if user_tags else 0

                # Apply tag matching
                filtered_data['tag_match_score'] = filtered_data['Tags'].apply(
                    lambda attraction_tags: calculate_tag_match(attraction_tags, tags)
                )

                # Keep only places with at least partial tag match
                filtered_data = filtered_data[filtered_data['tag_match_score'] > 0]
                filtered_data = filtered_data.sort_values('tag_match_score', ascending=False)

            # If no matches found, return empty DataFrame
            if filtered_data.empty:
                print(f"No recommendations found for province={province}, category={category}, tags={tags}")
                return pd.DataFrame()

            # Select and format results
            result = filtered_data.head(top_n)[['ID', 'Name', 'Province', 'Category', 'Tags']]
            
            # Format tags for display
            result['Tags'] = result['Tags'].apply(lambda x: ', '.join(x) if isinstance(x, list) else '')
            
            return result

        except Exception as e:
            import traceback
            print(f"Error in recommend method: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()

if __name__ == "__main__":
    print("ContentBasedRecommender is ready to use.")