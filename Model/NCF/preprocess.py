# preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import logging

def encode_features(df, user_encoder=None, item_encoder=None):
    """
    Encode categorical features like user IDs and item IDs.
    """
    try:
        if user_encoder is None:
            user_encoder = LabelEncoder()
        if item_encoder is None:
            item_encoder = LabelEncoder()

        if 'user_id' in df.columns:
            df['user_id'] = user_encoder.fit_transform(df['user_id'])
        if 'item_id' in df.columns:
            df['item_id'] = item_encoder.fit_transform(df['item_id'])
        return df, user_encoder, item_encoder
    except Exception as e:
        logging.error(f"Error encoding features: {e}")
        return df, user_encoder, item_encoder
