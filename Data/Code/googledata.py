import requests
import sqlite3
import time
import os

API_KEY = ""  # Set your Google Maps API key here or use an environment variable
PLACE_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"

# Dictionary of districts and their respective cities
districts_cities = {
    "Achham": ["Mangalsen", "Sanphebagar", "Kamalbazar", "Panchadewal Binayak", "Chaurpati", "Mellekh", "Dhakari", "Turmakhad", "Bannigadhi Jayagadh", "Ramaroshan"],
    "Arghakhanchi": ["Sandhikharka", "Sitganga", "Bhumikasthan", "Chhatradev", "Panini", "Malarani"],
    # ... (remaining districts) ...
    "Sunsari": ["Itahari", "Dharan", "Inaruwa", "Duhabi", "Ramdhuni", "Barah", "Dewangunj", "Koshi", "Gadhi", "Barju", "Bhokraha Narsingh"]
}

def fetch_places(query, next_page_token=None):
    params = {
        'input': query,
        'inputtype': 'textquery',
        'fields': 'formatted_address,name,rating,geometry',
        'key': API_KEY
    }
    if next_page_token:
        params['pagetoken'] = next_page_token

    try:
        response = requests.get(PLACE_SEARCH_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching places for query '{query}': {e}")
        return {}

def extract_tourist_places(data, district):
    """Extract relevant data from API response."""
    places = []
    candidates = data.get('candidates', [])
    print(f"Total candidates received: {len(candidates)}")
    for place in candidates:
        address = place.get('formatted_address', "")
        geometry = place.get('geometry', {}).get('location', {})
        address_parts = address.split(',')
        second_word_after_comma = address_parts[1].strip() if len(address_parts) > 1 else ""

        places.append({
            'name': place.get('name'),
            'address': address,
            'latitude': geometry.get('lat'),
            'longitude': geometry.get('lng'),
            'rating': place.get('rating'),
            'second_word': second_word_after_comma,
            'types': ', '.join(place.get('types', []))
        })
    print(f"Filtered places: {len(places)}")
    return places

def save_to_database(places):
    if not places:
        print("No places to save to the database.")
        return

    conn = sqlite3.connect('Dataset/tourist_attractions.db')
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attractions_nepal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            address TEXT,
            latitude REAL,
            longitude REAL,
            rating REAL,
            second_word TEXT,
            types TEXT,
            UNIQUE(name, latitude, longitude)
        )
    ''')

    for place in places:
        cursor.execute('''
            SELECT id, types FROM attractions_nepal
            WHERE name = ? AND latitude = ? AND longitude = ?
        ''', (place['name'], place['latitude'], place['longitude']))
        existing_entry = cursor.fetchone()

        if existing_entry:
            existing_id, existing_types = existing_entry
            if not existing_types or existing_types == 'N/A':
                cursor.execute('''
                    UPDATE attractions_nepal SET types = ? WHERE id = ?
                ''', (place.get('types', 'N/A'), existing_id))
            continue

        types = place.get('types', 'N/A')
        if types != 'N/A':
            types += f", {place['query_type']}"
        else:
            types = place['query_type']

        try:
            cursor.execute('''
                INSERT INTO attractions_nepal
                (name, address, latitude, longitude, rating, second_word, types)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (place['name'], place['address'], place['latitude'],
                  place['longitude'], place['rating'], place['second_word'], types))
        except sqlite3.IntegrityError as e:
            print(f"Error inserting {place['name']} - {place['address']}: {e}")

    conn.commit()
    conn.close()

types_keywords = [
    'park', 'amusement_park', 'church', 'tourist_attraction', 'mosque',
    'museum', 'zoo', 'aquarium', 'art_gallery', 'hindu_temple', 'monument',
    'historic_site', 'trek', 'picnic', 'hike'
]

def get_all_places(query, district):
    places = []
    next_page_token = None
    while True:
        print(f"Performing query: {query}")
        response_data = fetch_places(query, next_page_token)
        new_places = extract_tourist_places(response_data, district)
        places.extend(new_places)
        next_page_token = response_data.get('next_page_token')

        if not next_page_token:
            break

        print(f"Waiting for next page in district: {district}...")
        time.sleep(1)
    return places

for district, cities in districts_cities.items():
    all_places = []
    for city in cities:
        for kw in types_keywords:
            query_kw = f"{kw} in {city}, {district}, Nepal"
            places = get_all_places(query_kw, district)
            for place in places:
                place['query_type'] = kw
            all_places.extend(places)
    print(f"Fetched {len(all_places)} total places for district: {district}")
    save_to_database(all_places)
    print(f"Saved tourist places for district: {district}")
    time.sleep(2)

print("Data collection and saving process is complete.")
