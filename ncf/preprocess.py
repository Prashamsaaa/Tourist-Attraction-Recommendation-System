from sklearn.preprocessing import LabelEncoder
import pandas as pd

def load_and_preprocess_data(ratings_file, places_file):
    # Load CSV files
    ratings_df = pd.read_csv(ratings_file)
    attraction_df = pd.read_csv(places_file)
    
#    # Load datasets
#     attraction_df = pd.read_csv('C:/Users/dell/Desktop/major_project/project-final/Tourist-Attraction-Recommendation-System/Data/FinalDataset/Data.csv')
#     ratings_df = pd.read_csv('C:/Users/dell/Desktop/major_project/project-final/Tourist-Attraction-Recommendation-System/Data/FinalDataset/ratings.csv')
    
    # Data preprocessing
    ratings_df['id'] = ratings_df['id'].fillna(-1)
    ratings_df = ratings_df.drop_duplicates(subset=['user_id', 'id'])

    # Encode user_id and place_id
    user_encoder = LabelEncoder()
    place_encoder = LabelEncoder()
    ratings_df['user_id'] = user_encoder.fit_transform(ratings_df['user_id'])
    ratings_df['id'] = place_encoder.fit_transform(ratings_df['id'])

    num_users = len(user_encoder.classes_)
    num_items = len(place_encoder.classes_)
    print(num_users, num_items)

    ratings_df[['user_id','id']].to_csv(r'C:\Users\dell\Desktop\major_project\project-final\Tourist-Attraction-Recommendation-System\ncf\Data\processed.csv', index=False)

    
    
    return ratings_df, user_encoder, place_encoder