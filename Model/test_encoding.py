import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
import json

# Load Dataset
def load_data(file_path):
    if not file_path:
        raise FileNotFoundError("File path is invalid.")
    data = pd.read_csv(file_path)
    if 'Tags' not in data.columns or 'Province' not in data.columns:
        raise ValueError("Dataset must contain 'Tags' and 'Province' columns.")
    data['Tags'] = data['Tags'].apply(lambda x: x.split() if isinstance(x, str) else [])
    return data

# Load Categorized Tags
def load_categorized_tags(json_path):
    with open(json_path, 'r') as f:
        categorized_tags = json.load(f)
    return categorized_tags

# Determine Category Based on Tags
def determine_category(tags, categorized_tags):
    for category, category_tags in categorized_tags.items():
        if any(tag in category_tags for tag in tags):
            return category
    return 'Uncategorized'

# Create Feature Matrix
def create_feature_matrix(data, categorized_tags):
    # Assign categories to attractions based on their tags
    data['Category'] = data['Tags'].apply(lambda x: determine_category(x, categorized_tags))
    
    # Encode provinces and categories
    province_encoder = LabelEncoder()
    category_encoder = LabelEncoder()
    
    data['Encoded_Province'] = province_encoder.fit_transform(data['Province'])
    data['Encoded_Category'] = category_encoder.fit_transform(data['Category'])
    
    # Encode tags using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    tag_features = pd.DataFrame(mlb.fit_transform(data['Tags']), columns=mlb.classes_, index=data.index)
    
    # Combine all features into a single feature matrix
    feature_matrix = pd.concat([data[['Encoded_Province', 'Encoded_Category']], tag_features], axis=1)
    
    return feature_matrix, mlb, province_encoder, category_encoder

# Create User Profile
def create_user_profile(province, category, tags, feature_matrix, mlb, province_encoder, category_encoder):
    user_profile = pd.DataFrame(0, index=[0], columns=feature_matrix.columns)
    
    # Encode province and category into the user profile
    encoded_province = province_encoder.transform([province])[0]
    encoded_category = category_encoder.transform([category])[0]
    
    user_profile['Encoded_Province'] = encoded_province
    user_profile['Encoded_Category'] = encoded_category
    
    # Encode tags into the user profile
    for tag in tags:
        if tag in user_profile.columns:
            user_profile[tag] = 1
    
    return user_profile

# Recommend Attractions
def recommend_attractions(data, feature_matrix, user_profile, province, top_n=5):
    # Filter attractions by selected province
    filtered_data = data[data['Province'].str.strip().str.lower() == province.strip().lower()]
    
    if filtered_data.empty:
        print(f"No attractions found in the selected province: {province}")
        return pd.DataFrame()
    
    # Filter feature matrix based on filtered data indices
    filtered_feature_matrix = feature_matrix.loc[filtered_data.index]
    
    # Calculate cosine similarity between user profile and filtered attractions
    similarities = cosine_similarity(user_profile, filtered_feature_matrix)
    
    # Get top N recommendations based on similarity scores
    top_indices = np.argsort(similarities[0])[-top_n:][::-1]
    
    recommendations = filtered_data.iloc[top_indices][['ID', 'Name', 'Province', 'Tags']]
    
    return recommendations

# Display Options for User Input
def display_options(options_list, prompt):
    print(prompt)
    for i, option in enumerate(options_list, start=1):
        print(f"{i}. {option}")
    
    choice_index = int(input("\nEnter your choice (number): ")) - 1
    
    if choice_index < 0 or choice_index >= len(options_list):
        raise ValueError("Invalid choice. Please select a valid option.")
    
    return options_list[choice_index]

# Main Function to Run the Recommendation System with Options
def main():
    try:
        # Load dataset and categorized tags
        data_file_path = '../Notebook/Output/PreparedData.csv'
        categorized_tags_path = '../Data/FinalDataset/CategorizedTags.json'
        
        data = load_data(data_file_path)
        categorized_tags = load_categorized_tags(categorized_tags_path)
        
        # Create feature matrix and encoders
        feature_matrix, mlb, province_encoder, category_encoder = create_feature_matrix(data, categorized_tags)
        
        # Display available provinces for selection
        provinces = list(data['Province'].unique())
        selected_province = display_options(provinces, "Select a Province:")
        
        # Display available categories for selection
        categories = list(categorized_tags.keys())
        selected_category = display_options(categories, "Select a Category:")
        
        # Display available tags for the selected category and allow multiple selections
        available_tags = categorized_tags[selected_category]
        print("\nAvailable Tags:")
        for i, tag in enumerate(available_tags, start=1):
            print(f"{i}. {tag}")
        
        selected_tag_indices = input("\nEnter your preferred tags (comma-separated numbers): ").split(',')
        selected_tags = [available_tags[int(index.strip()) - 1] for index in selected_tag_indices]
        
        # Create user profile based on preferences
        user_profile = create_user_profile(selected_province, selected_category,
                                           selected_tags, feature_matrix,
                                           mlb, province_encoder,
                                           category_encoder)
        
        # Generate recommendations
        recommendations = recommend_attractions(data, feature_matrix,
                                                 user_profile,
                                                 selected_province,
                                                 top_n=5)
        
        if recommendations.empty:
            print("\nNo recommendations found.")
        else:
            print("\nRecommended Attractions:")
            print(recommendations.to_string(index=False))
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
