import torch
from recommendation import generate_recommendations
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from models import NCF
from recommendation import generate_recommendations
from config import *

model_path = './Model/models/model.pth'

# # Load the saved model directly using torch.load
# saved_state = torch.load('path_to_your_saved_model.pt')
# model = saved_state['model']
# place_encoder = saved_state['place_encoder_classes']

# # Set to evaluation mode
# model.eval()

attraction_df = pd.read_csv("./Data/FinalDataset/Data.csv")

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
model = torch.load(model_path, map_location='cpu')
place_encoder = LabelEncoder()
attraction_df['id'] = place_encoder.fit_transform(attraction_df['id'])
user_id =12        
ncf_recs = generate_recommendations(model, user_id, num_items= len(place_encoder.classes_),top_k =5, device = 'cpu', place_encoder=place_encoder, attraction_df= attraction_df)
print(ncf_recs)