import torch


def generate_recommendations(
    model,
    user_id,
    num_items,
    top_k,
    device,
    place_encoder,
    attraction_df,
    min_score=2.5,
):
    """
    Generates top-K recommendations for a user with confidence scores.
    """
    # Rename 'ID' to 'id' if needed
    if "ID" in attraction_df.columns and "id" not in attraction_df.columns:
        attraction_df.rename(columns={"ID": "id"}, inplace=True)

    print("Debug >> Attraction DataFrame columns:", attraction_df.columns)

    model.eval()
    with torch.no_grad():
        # Precompute all item IDs and user vector for performance
        user_vector = (
            torch.tensor([user_id], dtype=torch.long).repeat(num_items).to(device)
        )
        all_item_ids = torch.arange(num_items, dtype=torch.long).to(device)

        # Get predictions and confidence scores
        predictions = model(user_vector, all_item_ids).squeeze()

        # Get top-k recommendations, filtering by min_score
        scores, indices = torch.topk(predictions, k=min(top_k * 2, num_items))
        mask = scores >= min_score  # Apply minimum score filter
        filtered_indices = indices[mask][:top_k]
        filtered_scores = scores[mask][:top_k]

        recommended_items = filtered_indices.cpu().tolist()
        confidence_scores = filtered_scores.cpu().tolist()

        # Process recommendations
        recommendations = []

        # Create a mapping for item details retrieval (optimize for faster access)
        attraction_dict = attraction_df.set_index("id").to_dict(orient="index")

        for item_id, score in zip(recommended_items, confidence_scores):
            try:
                original_item_id = place_encoder.inverse_transform([item_id])[0]
                # Use the preprocessed attraction_dict for fast lookup
                attraction = attraction_dict.get(original_item_id)
                if attraction:
                    item_name = attraction.get(
                        "Name", attraction.get("name", "Unknown")
                    )
                    item_details = {
                        "id": original_item_id,
                        "Name": item_name,
                        "confidence": round(float(score), 3),
                    }

                    # Add additional details if available
                    for col in ["Tags", "rating", "Province"]:
                        if col in attraction:
                            item_details[col] = attraction[col]

                    recommendations.append(item_details)

            except Exception as e:
                print(f"Error processing item {item_id}: {str(e)}")
                continue

        return recommendations[:top_k]
