import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load the pre-trained GPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure the pad token is set (for padding if needed)
tokenizer.pad_token = tokenizer.eos_token

# Function to rewrite description using GPT-2
def rewrite_description_with_gpt2(description, name, location, place_type):
    """
    Use GPT-2 to rewrite a description to make it sound more formal and structured
    for a tourist site.
    """
    if not description:
        return ""
    
    # Prepare the input text for GPT-2
    input_text = f"Place: {name}\nLocation: {location}\nType: {place_type}\nExisting Description: {description}\n\nRewritten Description:"
    
    # Tokenize input text
    inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    
    # Create attention mask
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # 1 for actual tokens, 0 for padding
    
    # Generate the output with max_new_tokens to limit the total token count
    outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the output
    rewritten_description = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove the original input portion, keeping only the rewritten description
    rewritten_description = rewritten_description.replace(input_text, "").strip()
    
    return rewritten_description

# Main function to load CSV, process descriptions, and save the modified CSV
def clean_and_rewrite_places(input_file='places.csv', output_file='cleaned_places.csv'):
    # Load the CSV file into a DataFrame
    logging.info("Loading data from CSV...")
    df = pd.read_csv(input_file)
    
    # Check if the description column exists
    if 'description' not in df.columns:
        logging.error("Description column is missing.")
        return
    
    # Fill missing descriptions with empty strings
    logging.info("Filling missing descriptions with empty strings...")
    df['description'] = df['description'].fillna("")
    
    # Process descriptions in chunks (to avoid memory issues if the dataset is large)
    chunk_size = 5
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    
    logging.info(f"Processing {num_chunks} chunks of data...")
    all_rewritten_descriptions = []
    
    # Iterate over chunks of data
    for i in range(num_chunks):
        chunk = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        logging.info(f"Processing chunk {i + 1}/{num_chunks}...")
        
        # Rewrite descriptions for each chunk
        chunk['description'] = chunk.apply(lambda row: rewrite_description_with_gpt2(
            row['description'], row['name'], row['location'], row['types']), axis=1)
        
        # Save the chunk after processing
        chunk_output_file = f"cleaned_chunk_{i + 1}.csv"
        logging.info(f"Saving processed chunk {i + 1} to {chunk_output_file}...")
        chunk.to_csv(chunk_output_file, index=False)
        
        # Add the chunk to the list for concatenation
        all_rewritten_descriptions.append(chunk)
        
        # Optionally, add a small delay to avoid overwhelming the system
        time.sleep(1)
    
    # Concatenate all chunks back into one DataFrame
    logging.info("Concatenating all chunks...")
    df_cleaned = pd.concat(all_rewritten_descriptions, ignore_index=True)
    
    # Clean other columns (strip spaces)
    logging.info("Cleaning columns (stripping spaces)...")
    df_cleaned['name'] = df_cleaned['name'].str.strip()
    df_cleaned['address'] = df_cleaned['address'].str.strip()
    df_cleaned['location'] = df_cleaned['location'].str.strip()
    df_cleaned['types'] = df_cleaned['types'].str.strip()
    
    # Convert coordinates to numeric and drop rows with invalid coordinates
    logging.info("Converting coordinates to numeric values...")
    df_cleaned['latitude'] = pd.to_numeric(df_cleaned['latitude'], errors='coerce')
    df_cleaned['longitude'] = pd.to_numeric(df_cleaned['longitude'], errors='coerce')
    df_cleaned = df_cleaned.dropna(subset=['latitude', 'longitude'])
    
    # Remove rows with empty descriptions
    logging.info("Removing rows with empty descriptions...")
    df_cleaned = df_cleaned[df_cleaned['description'].str.len() > 0]
    
    # Save the final cleaned data to a new CSV file
    logging.info(f"Saving cleaned data to {output_file}...")
    df_cleaned.to_csv(output_file, index=False)
    
    logging.info(f"Original number of entries: {len(df)}")
    logging.info(f"Cleaned data saved to: {output_file}")
    
    # Print a sample of the cleaned descriptions
    logging.info("\nSample of cleaned descriptions:")
    sample = df_cleaned.sample(n=3)
    for _, row in sample.iterrows():
        logging.info(f"\nName: {row['name']}")
        logging.info(f"Description: {row['description']}")

# Run the script if this file is being executed directly
if __name__ == "__main__":
    clean_and_rewrite_places(input_file='FinalDataset/places.csv', output_file='cleaned_places.csv')
