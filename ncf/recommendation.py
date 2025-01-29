import torch
import heapq
import numpy as np

class Recommender:
    def __init__(self, model, item_mapping):
        self.model = model
        self.item_mapping = {v: k for k, v in item_mapping.items()}
        
    def get_recommendations(self, user_id, top_k=10):
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            user_input = torch.tensor([user_id] * len(self.item_mapping))
            item_input = torch.tensor(list(self.item_mapping.keys()))
            
            predictions = self.model(user_input, item_input)
            
        # Get top-k items
        top_k_items = heapq.nlargest(top_k, 
                                    range(len(predictions)),
                                    predictions.take)
        
        return [(self.item_mapping[idx], predictions[idx].item()) 
                for idx in top_k_items]
