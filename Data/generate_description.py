import os
import pandas as pd
import wikipedia
import warnings
from transformers import pipeline

# Suppress the GuessedAtParserWarning
warnings.filterwarnings("ignore", category=UserWarning, module="wikipedia")

# Initialize text generation model (Hugging Face pipeline)
generator = pipeline('text-generation', model='gpt2')

# Function to get Wikipedia summary or specific sections
def get_wikipedia_summary(place_name, location, types):
    try:
        # Search for the page using the place name, location, and type
        search_query = f"{place_name} {types} in {location}"
        print(f"Searching for: {search_query}")
        
        # Perform a search query on Wikipedia to find matching pages
        search_results = wikipedia.search(search_query)

        if not search_results:
            print(f"No results found for {search_query}")
            return "No results found in Wikipedia for this location."

        # Attempt to retrieve the page summary of the first result
        page = wikipedia.page(search_results[0])

        # Try fetching specific sections if available
        sections = ["History", "Attractions", "Significance", "Culture", "Geography", "Tourism"]
        for section in sections:
            try:
                section_content = page.section(section)
                if section_content:
                    print(f"Found section: {section}")
                    return section_content[:600]  # Limit to first 600 characters of section
            except Exception as e:
                print(f"Error fetching section {section}: {e}")
                pass  # Continue if a section is not found

        # Fallback to full summary if no section found
        print(f"Returning full summary for {place_name}")
        return page.summary[:600]  # Limit to first 600 characters of the summary
    
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for {place_name}: {e.options[:5]}")
        return f"Multiple results found. Try specifying the place more clearly. Options: {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        print(f"Page error for {place_name}")
        return "No Wikipedia page found for this location."
    except Exception as e:
        print(f"Error with Wikipedia fetch for {place_name}: {str(e)}")
        return f"Error: {str(e)}"

# Function to generate a detailed description, using Wikipedia first, then AI generation as fallback
def generate_actual_description(place_name, location, types):
    wikipedia_description = get_wikipedia_summary(place_name, location, types)
    if wikipedia_description and not wikipedia_description.startswith("Multiple results found"):
        print(f"Using Wikipedia description for {place_name}")
        return wikipedia_description  # Return Wikipedia description if available
    
    # Fallback to AI-based description generation if Wikipedia summary is not found
    print(f"Using AI-generated description for {place_name}")
    prompt = (f"Provide a detailed, informative, and engaging description of the place '{place_name}', a {types} in {location}. "
              "Describe its historical significance, cultural importance, key attractions, famous landmarks, "
              "natural beauty, any famous events or people associated with it, and its impact on the local community. "
              "Provide a well-rounded perspective on the location that would be informative to a global audience.")
    
    # Generate a longer description with GPT-2
    response = generator(prompt, max_length=300, num_return_sequences=1, truncation=True)
    return response[0]['generated_text']

# Function to process the CSV file in chunks
def process_large_csv(input_csv, output_csv, chunk_size=5):
    # Check if the output CSV exists
    file_exists = os.path.isfile(output_csv)

    # Process the input CSV in chunks
    for chunk in pd.read_csv(input_csv, chunksize=chunk_size):
        print(f"Processing chunk with {len(chunk)} rows...")

        # Convert all column names to lowercase
        chunk.columns = chunk.columns.str.lower()

        # Generate descriptions for each row in the chunk
        chunk['description'] = chunk.apply(
            lambda row: generate_actual_description(row['name'], row['address'], row['types']), axis=1
        )

        # Append the chunk with descriptions to the output CSV
        chunk.to_csv(output_csv, mode='a', index=False, header=not file_exists)

        # After writing, set file_exists to True to prevent writing headers on subsequent chunks
        file_exists = True

    print(f"Descriptions added to '{output_csv}'.")

# Input and output file paths
input_csv = "FinalDataset/attractions_final.csv"  # Replace with the path to your input CSV file
output_csv = "FinalDataset/places_with_descriptions.csv"  # Replace with the path to your output CSV file

# Process the CSV file and generate descriptions
process_large_csv(input_csv, output_csv)
