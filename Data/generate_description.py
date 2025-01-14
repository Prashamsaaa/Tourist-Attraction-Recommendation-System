import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import random
import logging
import os
import re

def setup_logging():
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/scraping_{time.strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

class PlaceScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5'
        }
        self.session = requests.Session()

    def get_description_from_wikipedia(self, name: str) -> str:
        """Get description from Wikipedia based on the place name."""
        try:
            search_query = f"{name} site:en.wikipedia.org"
            search_url = f"https://www.google.com/search?q={requests.utils.quote(search_query)}"
            response = self.session.get(search_url, headers=self.headers, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')
            link = soup.find('a', href=re.compile('^/url\?q=https://en.wikipedia.org/wiki/'))
            
            if link:
                wikipedia_url = link['href'].split('q=')[1]
                wikipedia_response = requests.get(wikipedia_url, headers=self.headers)
                wikipedia_soup = BeautifulSoup(wikipedia_response.text, 'html.parser')

                # Extract description from the first paragraph
                paragraphs = wikipedia_soup.find_all('p')
                description = ""
                for para in paragraphs:
                    description += para.get_text()
                    if len(description) > 200:  # Ensure it's long enough to be a valid description
                        break
                
                return description.strip()
        except Exception as e:
            logging.error(f"Error getting description from Wikipedia: {str(e)}")
        return None

    def get_description(self, name: str) -> dict:
        """Get description from Wikipedia."""
        try:
            time.sleep(random.uniform(2, 4))
            result = {'description': None, 'website': None}

            # Try Wikipedia first
            description = self.get_description_from_wikipedia(name)
            if description:
                result['description'] = description
                result['website'] = "Wikipedia"

            return result

        except Exception as e:
            logging.error(f"Error scraping {name}: {str(e)}")
            return {'description': None, 'website': None}

def process_attractions(input_file: str, output_file: str, no_description_file: str):
    """Process the attractions CSV file and filter out places with no description"""
    try:
        # Read input file
        logging.info(f"Reading input file: {input_file}")
        df = pd.read_csv(input_file)
        original_count = len(df)
        logging.info(f"Found {original_count} entries to process")
        
        # Initialize scraper
        scraper = PlaceScraper()
        
        # Add new columns if they don't exist
        if 'description' not in df.columns:
            df['description'] = None
        if 'website' not in df.columns:
            df['website'] = None
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        os.makedirs(os.path.dirname(no_description_file), exist_ok=True)
        
        # Prepare output files for descriptions and no descriptions
        df_no_description = pd.DataFrame(columns=df.columns)
        
        # Process each place
        for index, row in df.iterrows():
            try:
                logging.info(f"Processing {index + 1}/{original_count}: {row['name']}")
                
                # Only scrape if we don't have a description yet
                if pd.isna(df.at[index, 'description']) or df.at[index, 'description'] == '':
                    result = scraper.get_description(row['name'])
                    
                    # Update dataframe with results
                    if result['description']:
                        df.at[index, 'description'] = result['description']
                        df.at[index, 'website'] = result['website']
                    else:
                        # Append rows with no description to a separate dataframe
                        df_no_description = pd.concat([df_no_description, df.iloc[[index]]], ignore_index=True)
                
                # Save progress every 5 entries
                if (index + 1) % 5 == 0:
                    df.to_csv(output_file, index=False, encoding='utf-8')
                    logging.info(f"Progress saved: {index + 1}/{original_count} entries processed")
                
            except Exception as e:
                logging.error(f"Error processing row {index}: {str(e)}")
                continue
        
        # Save final CSV with descriptions
        df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"Descriptions saved to {output_file}")
        
        # Save CSV for places without descriptions
        df_no_description.to_csv(no_description_file, index=False, encoding='utf-8')
        logging.info(f"Rows without descriptions saved to {no_description_file}")
        
        # Log completion statistics
        final_descriptions = df['description'].notna().sum()
        logging.info(f"Processing complete! Added descriptions for {final_descriptions} out of {original_count} places")
        
    except Exception as e:
        logging.error(f"Fatal error: {str(e)}")
        raise

if __name__ == "__main__":
    # Set up logging
    setup_logging()
    
    # Define file paths
    input_file = "FinalDataset/attractions_without_description.csv"
    output_file = "FinalDataset/enriched_attractions.csv"
    no_description_file = "FinalDataset/no_description_attractions.csv"
    
    # Process the file
    process_attractions(input_file, output_file, no_description_file)
