import pandas as pd
import time
from groq import Groq

def generate_place_description(client, place_name):
    """
    Generate a description for a place using Groq API
    """
    prompt = f"Generate a brief description (2-3 sentences) of {place_name}. Focus on its key features and significance."
    
    try:
        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="mixtral-8x7b-32768",  # You can change the model as needed
            temperature=0.7,
            max_tokens=150
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Error generating description for {place_name}: {str(e)}")
        return "Description generation failed"

def process_places_csv(input_file, output_file, api_key):
    """
    Process places from input CSV and generate descriptions using pandas
    """
    # Initialize Groq client
    client = Groq(api_key=api_key)
    
    # Read input CSV
    try:
        # Read input CSV first without dtype specification
        df = pd.read_csv(input_file)
        
        # Clean and convert id column
        df['id'] = pd.to_numeric(df['id'], errors='coerce')  # Convert to numeric, invalid values become NaN
        df = df.dropna(subset=['id'])  # Remove rows with NaN in id column
        df['id'] = df['id'].astype(int)  # Convert to integer
        
        # Validate required columns
        required_columns = ['id', 'name']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Input CSV must contain columns: {required_columns}")
        
        # Generate descriptions
        print("Generating descriptions...")
        descriptions = []
        for place_name in df['name']:
            print(f"Processing: {place_name}")
            description = generate_place_description(client, place_name)
            descriptions.append(description)
            time.sleep(1)  # Rate limiting
        
        # Add descriptions to dataframe
        df['description'] = descriptions
        
        # Save to output CSV
        df.to_csv(output_file, index=False)
        return True
        
    except Exception as e:
        print(f"Error processing CSV: {str(e)}")
        return False

def main():
    # Configuration
    INPUT_CSV = 'lastremaining.csv'  # Input CSV with columns: id, name
    OUTPUT_CSV = 'lastremaining_output.csv'  # Output CSV with columns: id, name, description
    GROQ_API_KEY = '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'  # Replace with your actual Groq API key
    
    if process_places_csv(INPUT_CSV, OUTPUT_CSV, GROQ_API_KEY):
        print(f"\nProcess completed successfully! Results saved to {OUTPUT_CSV}")
    else:
        print("\nProcess failed. Please check the error messages above.")

if __name__ == "__main__":
    main()