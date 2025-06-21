import torch

def generate_recommendations(
    model,
    user_id,
    merged_df,
    top_k,
    device,
    place_encoder,
    min_score=2.5,
):
    """
    Generates top-K recommendations for a user with confidence scores using the merged dataset.
    """
    if "ID" in merged_df.columns and "id" not in merged_df.columns:
        merged_df.rename(columns={"ID": "id"}, inplace=True)
    
    print("Debug >> Merged DataFrame columns:", merged_df.columns)

    model.eval()
    with torch.no_grad():
        num_items = len(place_encoder.classes_)
        user_vector = torch.tensor([user_id], dtype=torch.long).repeat(num_items).to(device)
        all_item_ids = torch.arange(num_items, dtype=torch.long).to(device)
        
        predictions = model(user_vector, all_item_ids).squeeze()
        scores, indices = torch.topk(predictions, k=min(top_k * 2, num_items))
        mask = scores >= min_score
        filtered_indices = indices[mask][:top_k]
        filtered_scores = scores[mask][:top_k]
        
        recommended_items = filtered_indices.cpu().tolist()
        confidence_scores = filtered_scores.cpu().tolist()
        
        recommendations = []
        
        # Aggregate the ratings for each attraction
        merged_df = merged_df.groupby('id').agg({
            'name': 'first',  # Keep the first name
            'description': 'first',  # Keep the first description
            'province': 'first',  # Keep the first province
            'tags': 'first',  # Keep the first tags
            'rating': 'mean'  # Average ratings
        }).reset_index()

        # Convert to dictionary with unique 'id'
        attraction_dict = merged_df.set_index("id").to_dict(orient="index")
        
        for item_id, score in zip(recommended_items, confidence_scores):
            try:
                original_item_id = place_encoder.inverse_transform([item_id])[0]
                attraction = attraction_dict.get(original_item_id)
                if attraction:
                    item_name = attraction.get("name", "Unknown")
                    item_details = {
                        "id": original_item_id,
                        "name": item_name,
                        "confidence": round(float(score), 3),
                    }
                    for col in ["Tags", "Rating", "Province"]:
                        if col in attraction:
                            item_details[col] = attraction[col]
                    recommendations.append(item_details)
            except Exception as e:
                print(f"Error processing item {item_id}: {str(e)}")
                continue
        
        return recommendations[:top_k]
