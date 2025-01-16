import csv
import requests

# Function to fetch description from Wikipedia
def fetch_description_from_wikipedia(city):
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city.replace(' ', '_')}"
    try:
        response = requests.get(url)
        data = response.json()
        if "extract" in data:
            return data["extract"]
        else:
            return None
    except Exception as e:
        print(f"Error fetching description for {city} from Wikipedia: {e}")
        return None

# Function to search on Google if Wikipedia does not provide a description
def fetch_description_from_google(city, api_key, cse_id):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={city}&key={api_key}&cx={cse_id}"
    try:
        response = requests.get(search_url)
        data = response.json()
        if "items" in data:
            return data["items"][0]["snippet"]  # Fetch the snippet from the first result
        else:
            return f"No information found on Google for {city}."
    except Exception as e:
        print(f"Error fetching Google results for {city}: {e}")
        return f"No information found for {city}."

# Function to get the description from either Wikipedia or Google
def fetch_description(city, api_key, cse_id):
    description = fetch_description_from_wikipedia(city)
    if description:
        return description
    else:
        print(f"Wikipedia not found for {city}, searching Google...")
        return fetch_description_from_google(city, api_key, cse_id)

# Process the CSV file with city names
def process_city_list(input_file, output_file, api_key, cse_id):
    with open(input_file, mode='r', newline='', encoding='utf-8-sig') as infile, open(output_file, mode='w', newline='', encoding='utf-8-sig') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write header for the new CSV
        writer.writerow(["City", "Description"])

        # Process each row (each row contains only a city name)
        for row in reader:
            # Skip empty rows or rows with only whitespace
            if not row or not row[0].strip():
                continue

            city = row[0].strip()  # Get the city name and remove extra spaces
            description = fetch_description(city, api_key, cse_id)
            writer.writerow([city, description])
            print(f"Processed: {city}")

    print(f"Descriptions added and saved to '{output_file}'")

# Input and output file paths
input_csv = "FinalDataset/check.csv"  # Replace with your input file path
output_csv = "cities_with_descriptions.csv"  # Replace with your desired output file path

# Set your Google Custom Search API credentials
api_key = "AIzaSyBffzHsIddLw-0FpZqU9lgBvajDql7qSlg"  # Replace with your Google API key
cse_id = "7733fac50010548ea"  # Replace with your Custom Search Engine ID

# Process the file
process_city_list(input_csv, output_csv, api_key, cse_id)
