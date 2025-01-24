import pandas as pd
from transformers import pipeline
import torch

# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1  # Use GPU if available, otherwise use CPU

# Initialize the zero-shot classification pipeline from Hugging Face
classifier = pipeline("zero-shot-classification", device=device)

# Define the possible candidate labels (generic types)
candidate_labels = [
    'park', 'temple', 'museum', 'monument', 'garden', 'beach', 'lake', 'mountain', 'market', 'zoo', 
    'church', 'mosque', 'tower', 'aquarium', 'palace', 'fort', 'ruins', 'picnic', 'waterfall', 
    'forest', 'river', 'art_gallery', 'historical_site', 'archeological_site', 'hiking', 'trek'
]

# Function to classify the description and infer the type
def get_type_from_huggingface(description, row_index):
    try:
        # Show the current row index and description being processed
        print(f"Processing row {row_index}: {description[:100]}...")  # Show only first 100 chars for brevity
        result = classifier(description, candidate_labels)
        return result['labels'][0]  # Return the most probable label
    except Exception as e:
        print(f"Error processing row {row_index}: {e}")
        return 'other'

# Load the CSV file with descriptions
input_file = 'manual.csv'  # Replace with your CSV file name
df = pd.read_csv(input_file)

# Ensure the 'types' and 'description' columns exist
if 'types' not in df.columns or 'description' not in df.columns:
    print("Error: 'types' or 'description' column not found in the CSV file.")
else:
    processed_count = 0

    # Iterate through each row and update the type if a better match is found
    for index, row in df.iterrows():
        current_type = str(row['types']).lower()
        description = str(row['description'])
        
        # Only update the type if it is 'other'
        if current_type == 'other':
            inferred_type = get_type_from_huggingface(description, index)

            # Update the type if a better match is found
            df.at[index, 'types'] = inferred_type

        processed_count += 1

        # Save the updated data every 5 rows based on processed_count
        if processed_count % 5 == 0:
            output_file = 'updated_file.csv'  # Replace with the desired output file name
            df.to_csv(output_file, index=False)
            print(f"Progress saved after {processed_count} rows to '{output_file}'.")

    # Final save
    output_file = 'updated_file.csv'  # Replace with the desired output file name
    df.to_csv(output_file, index=False)
    print(f"Final updated data saved to '{output_file}'")
