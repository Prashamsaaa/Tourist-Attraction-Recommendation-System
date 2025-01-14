import csv
import os
import requests
from bs4 import BeautifulSoup
import time

def get_description_from_url(url):
    """
    Scrapes the description of a place from the given URL.
    :param url: URL to scrape the description from.
    :return: A description of the place if found, else None.
    """
    try:
        # Send GET request to the URL
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the description from the page content (customize based on actual HTML structure)
        description_tag = soup.find('meta', {'name': 'description'})
        
        if description_tag:
            description = description_tag.get('content', None)
        else:
            # For some websites, description may be in a specific paragraph or div
            description = soup.find('p')  # Adjust this to target the correct element if needed
            description = description.get_text() if description else None

        return description
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return None

def process_csv(input_file, found_file, not_found_file, save_interval=5):
    """
    Reads a CSV file, fetches descriptions by scraping URLs, and writes results to separate files.
    Saves progress after every 'save_interval' places.
    :param input_file: Path to the input CSV file.
    :param found_file: Path to the output file for places with descriptions.
    :param not_found_file: Path to the output file for places without descriptions.
    :param save_interval: Number of rows to process before saving progress.
    """
    # Load existing progress if files already exist
    processed_ids = set()
    if os.path.exists(found_file):
        with open(found_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            processed_ids.update(row["id"] for row in reader)

    if os.path.exists(not_found_file):
        with open(not_found_file, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            processed_ids.update(row["id"] for row in reader)

    with open(input_file, mode='r', encoding='utf-8') as infile, \
         open(found_file, mode='a', encoding='utf-8', newline='') as found_outfile, \
         open(not_found_file, mode='a', encoding='utf-8', newline='') as not_found_outfile:
        
        reader = csv.DictReader(infile)
        found_writer = csv.DictWriter(found_outfile, fieldnames=reader.fieldnames + ["description"])
        not_found_writer = csv.DictWriter(not_found_outfile, fieldnames=reader.fieldnames)
        
        # Write headers if files are new
        if os.stat(found_file).st_size == 0:
            found_writer.writeheader()
        if os.stat(not_found_file).st_size == 0:
            not_found_writer.writeheader()
        
        count = 0
        for row in reader:
            if row["id"] in processed_ids:
                continue  # Skip already processed rows

            url = row["url"]  # URL column from your CSV
            print(f"Fetching description for: {url}")
            
            description = get_description_from_url(url)
            
            if description:
                row["description"] = description
                found_writer.writerow(row)
            else:
                not_found_writer.writerow(row)

            count += 1
            processed_ids.add(row["id"])

            # Save progress every 'save_interval' rows
            if count % save_interval == 0:
                print(f"Saved progress after processing {count} places.")
                # Save progress after every save_interval
                found_outfile.flush()
                not_found_outfile.flush()

        print("Processing complete!")

# Example Usage
input_csv = "Data/FinalDataset/attractions_with_links.csv"       # Input file path with links
found_csv = "Data/FinalDataset/places_with_description.csv"  # Output file for found descriptions
not_found_csv = "Data/FinalDataset/places_without_description.csv"  # Output file for not found

process_csv(input_csv, found_csv, not_found_csv, save_interval=5)
