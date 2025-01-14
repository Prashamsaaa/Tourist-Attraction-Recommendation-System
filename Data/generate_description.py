import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

# Load pre-trained GPT-2 model and tokenizer from Hugging Face
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Ensure pad token is defined
tokenizer.pad_token = tokenizer.eos_token

def clean_and_rewrite_places(input_file='FinalDataset/places.csv', output_file='cleaned_places.csv'):
    # Read the CSV file
    df = pd.read_csv(input_file)
    
    # Check if the column name is misspelled (e.g., 'escription' instead of 'description')
    if 'escription' in df.columns:
        df.rename(columns={'escription': 'description'}, inplace=True)
    
    if 'description' not in df.columns:
        print("Description column is missing.")
        return
    
    # Fill missing descriptions with empty strings
    df['description'] = df['description'].fillna("")
    
    def rewrite_description_with_gpt2(description, name, location, place_type):
        """
        Use GPT-2 to rewrite a description to make it more descriptive and less review-like.
        """
        if not description:
            return ""
        
        # Prepare input text for GPT-2
        input_text = f"Place: {name}\nLocation: {location}\nType: {place_type}\nExisting Description: {description}\n\nRewritten Description:"
        
        # Tokenize input with attention mask
        inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
        
        # Create an attention mask to avoid padding issues
        attention_mask = torch.ones(inputs.shape, dtype=torch.long)  # 1 for actual tokens, 0 for padding
        
        # Generate the output with max_new_tokens to prevent total length overflow
        outputs = model.generate(inputs, attention_mask=attention_mask, max_new_tokens=150, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7, pad_token_id=tokenizer.eos_token_id)
        
        # Decode the generated text
        rewritten_description = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original input part to leave only the rewritten description
        rewritten_description = rewritten_description.replace(input_text, "").strip()
        
        return rewritten_description

    # Split the DataFrame into chunks of 5 rows
    chunk_size = 5
    num_chunks = len(df) // chunk_size + (1 if len(df) % chunk_size != 0 else 0)
    
    all_rewritten_descriptions = []
    
    # Iterate over the chunks
    print("Processing descriptions in chunks of 5...")
    for i in range(num_chunks):
        chunk = df.iloc[i*chunk_size : (i+1)*chunk_size]
        
        # Rewrite descriptions for each chunk
        chunk['description'] = chunk.apply(lambda row: rewrite_description_with_gpt2(row['description'], row['name'], row['location'], row['types']), axis=1)
        
        # Append the rewritten chunk to the final list
        all_rewritten_descriptions.append(chunk)
        
        # Optional: Introduce a small delay to avoid overloading the system
        time.sleep(1)
    
    # Concatenate all chunks back into a single DataFrame
    df_cleaned = pd.concat(all_rewritten_descriptions, ignore_index=True)
    
    # Clean other columns (strip spaces)
    df_cleaned['name'] = df_cleaned['name'].str.strip()
    df_cleaned['address'] = df_cleaned['address'].str.strip()
    df_cleaned['location'] = df_cleaned['location'].str.strip()
    df_cleaned['types'] = df_cleaned['types'].str.strip()
    
    # Convert coordinates to numeric, replacing invalid values with NaN
    df_cleaned['latitude'] = pd.to_numeric(df_cleaned['latitude'], errors='coerce')
    df_cleaned['longitude'] = pd.to_numeric(df_cleaned['longitude'], errors='coerce')
    
    # Remove rows with invalid coordinates
    df_cleaned = df_cleaned.dropna(subset=['latitude', 'longitude'])
    
    # Remove rows with empty descriptions
    df_cleaned = df_cleaned[df_cleaned['description'].str.len() > 0]
    
    # Save to CSV
    df_cleaned.to_csv(output_file, index=False)
    
    print(f"Original number of entries: {len(df)}")
    print(f"Cleaned data saved to: {output_file}")
    
    # Print sample of cleaned descriptions
    print("\nSample of cleaned descriptions:")
    sample = df_cleaned.sample(n=3)
    for _, row in sample.iterrows():
        print(f"\nName: {row['name']}")
        print(f"Description: {row['description']}")

if __name__ == "__main__":
    clean_and_rewrite_places()
