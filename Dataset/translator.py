import pandas as pd
import os

# Load the CSV file
file_path = 'Dataset/output/attractions-nepal-uncleaned.csv'  # Update with your input file path
data = pd.read_csv(file_path)

# Create a transliteration mapping for Nepali characters
nepali_to_roman_map = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ii', 'उ': 'u', 'ऊ': 'uu', 'ऋ': 'ri',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'अं': 'am', 'अः': 'ah',
    'क': 'ka', 'ख': 'kha', 'ग': 'ga', 'घ': 'gha', 'ङ': 'nga',
    'च': 'cha', 'छ': 'chha', 'ज': 'ja', 'झ': 'jha', 'ञ': 'nya',
    'ट': 'ta', 'ठ': 'tha', 'ड': 'da', 'ढ': 'dha', 'ण': 'na',
    'त': 'ta', 'थ': 'tha', 'द': 'da', 'ध': 'dha', 'न': 'na',
    'प': 'pa', 'फ': 'pha', 'ब': 'ba', 'भ': 'bha', 'म': 'ma',
    'य': 'ya', 'र': 'ra', 'ल': 'la', 'व': 'wa', 'श': 'sha',
    'ष': 'sha', 'स': 'sa', 'ह': 'ha',
    'ा': 'a', 'ि': 'i', 'ी': 'ii', 'ु': 'u', 'ू': 'uu',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ं': 'm', 'ः': 'h',
    '्': '', ' ': ' '  # Halant and space
}

# Transliteration function
def transliterate(text):
    transliterated = ""
    for char in text:
        transliterated += nepali_to_roman_map.get(char, char)  # Default to original if no mapping exists
    return transliterated

# Incremental saving
output_path = 'Dataset/output/output.csv'  # Update with your output file path

# Check if the output file already exists and remove it
if os.path.exists(output_path):
    os.remove(output_path)

# Process rows and save incrementally
batch_size = 10
rows = []

for index, row in data.iterrows():
    for column in ['name', 'address', 'second_word']:
        if pd.notnull(row[column]) and any('\u0900' <= c <= '\u097F' for c in str(row[column])):
            row[column] = transliterate(row[column])  # Transliterate if Nepali text is detected
    rows.append(row)

    # Save in batches
    if (index + 1) % batch_size == 0 or index == len(data) - 1:
        batch_df = pd.DataFrame(rows)
        if os.path.exists(output_path):
            batch_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            batch_df.to_csv(output_path, mode='w', header=True, index=False)
        rows = []  # Clear the batch

print(f"Transliteration complete. Results saved incrementally to {output_path}")
