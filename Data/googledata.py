import requests
import sqlite3
import time
import os

# Replace with your API key
API_KEY = "dfjhsgfsdjhfgfdfdsfsdsf"
API_KEY 

# Define the API endpoint
PLACE_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json"

# Dictionary of districts and their respective cities
districts_cities = {
    "Achham": ["Mangalsen", "Sanphebagar", "Kamalbazar", "Panchadewal Binayak", "Chaurpati", "Mellekh", "Dhakari", "Turmakhad", "Bannigadhi Jayagadh", "Ramaroshan"],
    "Arghakhanchi": ["Sandhikharka", "Sitganga", "Bhumikasthan", "Chhatradev", "Panini", "Malarani"],
    "Baglung": ["Baglung", "Galkot", "Jaimuni", "Dhorpatan", "Bareng", "Kathekhola", "Tamankhola", "Tara Khola", "Nisikhola", "Badigad"],
    "Baitadi": ["Dasharathchand", "Patan", "Melauli", "Purchaudi", "Sunarya", "Sigas", "Shivanath", "Pancheshwar", "Dilasaini", "Dogdakedar"],
    "Bajhang": ["Jayaprithvi", "Bungal", "Talkot", "Masta", "Khaptad Chhanna", "Thalara", "Bitthadchir", "Surma", "Chhabis Pathibhera", "Durgathali"],
    "Bajura": ["Budhinanda", "Triveni", "Budhiganga", "Gaumul", "Pandav Gupha", "Swamikartik", "Chhededaha", "Himali", "Jagannath"],
    "Banke": ["Nepalgunj", "Kohalpur", "Narainapur", "Rapti Sonari", "Baijanath", "Khajura", "Duduwa", "Janaki"],
    "Bara": ["Kalaiya", "Jitpur Simara", "Kolhabi", "Nijgadh", "Mahagadhimai", "Simraungadh", "Pacharauta", "Pheta", "Bishrampur"],
    "Bardiya": ["Gulariya", "Madhuwan", "Rajapur", "Thakurbaba", "Basgadhi", "Barbardiya", "Badhaiyatal", "Geruwa"],
    "Bhaktapur": ["Bhaktapur", "Madhyapur Thimi", "Changunarayan", "Suryabinayak"],
    "Bhojpur": ["Bhojpur", "Shadananda", "Temkemaiyung", "Ramprasad Rai", "Hatuwagadhi", "Pauwadungma", "Salpasilichho", "Arun", "Aamchowk"],
    "Chitwan": ["Bharatpur", "Ratnanagar", "Kalika", "Rapti", "Khairahani", "Madi", "Ichchhakamana"],
    "Dadeldhura": ["Amargadhi", "Parshuram", "Aalital", "Bhageshwar", "Navadurga", "Ajayameru", "Ganyapdhura"],
    "Dailekh": ["Narayan", "Dullu", "Aathbis", "Bhairabi", "Mahabu", "Gurans", "Thantikandh", "Bhagawatimai", "Dungeshwar", "Naumule", "Chamunda Bindrasaini"],
    "Dang": ["Tulsipur", "Ghorahi", "Lamahi", "Banglachuli", "Dangisharan", "Gadhawa", "Rajpur", "Rapti", "Shantinagar", "Babai"],
    "Darchula": ["Mahakali", "Shailyashikhar", "Malikarjun", "Apihimal", "Naugad", "Duhu", "Marma", "Lekam", "Vyans"],
    "Dhading": ["Dhunibesi", "Nilkantha", "Siddhalek", "Khaniyabas", "Gangajamuna", "Galchi", "Thakre", "Netrawati Dabjong", "Benighat Rorang", "Gajuri", "Tripurasundari", "Rubi Valley"],
    "Dhankuta": ["Dhankuta", "Mahalaxmi", "Pakhribas", "Sangurigadhi", "Chhathar Jorpati", "Chaubise", "Sahidbhumi"],
    "Dhanusa": ["Janakpur", "Chhireshwarnath", "Ganeshman Charnath", "Dhanushadham", "Nagarain", "Hansapur", "Sabila", "Mithila Bihari"],
    "Dolakha": ["Bhimeshwar", "Jiri", "Kalinchok", "Melung", "Bigu", "Gaurishankar", "Baiteshwar", "Sailung", "Tamakoshi"],
    "Dolpa": ["Dunai", "Thuli Bheri", "Tripurasundari", "Dolpo Buddha", "She Phoksundo", "Jagadulla", "Mudkechula", "Kaike"],
    "Doti": ["Dipayal Silgadhi", "Shikhar", "Purbichauki", "Badikedar", "Jorayal", "Sayal", "Aadarsha", "K I Singh", "Bogtan Fudsil"],
    "Gorkha": ["Gorkha", "Palungtar", "Sulikot", "Siranchok", "Ajirkot", "Gandaki", "Bhimsen", "Sahid Lakhan", "Dharche", "Aarughat", "Chum Nubri"],
    "Gulmi": ["Resunga", "Musikot", "Isma", "Kaligandaki", "Gulmidarbar", "Satyawati", "Chandrakot", "Ruru", "Chatrakot", "Dhurkot", "Madane", "Malika"],
    "Humla": ["Simkot", "Namkha", "Sarkegad", "Adanchuli", "Kharpunath", "Chankheli", "Tanjakot"],
    "Ilam": ["Ilam", "Deumai", "Suryodaya", "Mai", "Sandakpur", "Rong", "Mangsebung", "Chulachuli", "Fakphokthum", "Mai Jogmai"],
    "Jajarkot": ["Bheri", "Chhedagad", "Junichande", "Barekot", "Kuse", "Shivalaya", "Nalgad"],
    "Jhapa": ["Mechinagar", "Birtamod", "Damak", "Kankai", "Bhadrapur", "Arjundhara", "Shivasatakshi", "Gauradaha", "Buddhashanti", "Haldibari"],
    "Jumla": ["Chandannath", "Kankasundari", "Sinja", "Hima", "Tila", "Guthichaur", "Tatopani", "Patarasi"],
    "Kailali": ["Dhangadhi", "Tikapur", "Lamki Chuha", "Ghodaghodi", "Bhajani", "Godawari", "Gauriganga", "Janaki", "Bardgoriya", "Mohanyal", "Chure", "Kailari", "Joshipur"],
    "Kalikot": ["Khandachakra", "Raskot", "Tilagufa", "Pachaljharana", "Sanni Triveni", "Narharinath", "Shubha Kalika", "Mahawai", "Palata"],
    "Kanchanpur": ["Bhimdatta", "Punarbas", "Bedkot", "Belauri", "Krishnapur", "Mahakali", "Shuklaphanta", "Laljhadi", "Beldandi"],
    "Kapilvastu": ["Kapilvastu", "Banganga", "Shivaraj", "Krishnanagar", "Maharajganj", "Buddhabhumi", "Yasodhara", "Mayadevi", "Suddhodhan", "Bijaynagar"],
    "Kaski": ["Pokhara", "Annapurna", "Machhapuchchhre", "Madi", "Rupa"],
    "Kathmandu": ["Kathmandu", "Budhanilkantha", "Tokha", "Tarakeshwar", "Dakshinkali", "Nagarjun", "Kageshwori-Manohara", "Gokarneshwor", "Chandragiri", "Kirtipur", "Shankharapur"],
    "Kavrepalanchok": ["Dhulikhel", "Banepa", "Panauti", "Panchkhal", "Namobuddha", "Mandandeupur", "Roshi", "Temal", "Bethanchowk", "Khanikhola"],
    "Khotang": ["Diktel Rupakot Majhuwagadhi", "Halesi Tuwachung", "Khotehang", "Diprung", "Aiselukharka", "Jantedhunga", "Kepilasgadhi", "Barahapokhari"],
    "Lalitpur": ["Lalitpur", "Godawari", "Mahalaxmi", "Konjyosom", "Bagmati"],
    "Lamjung": ["Besisahar", "Madhya Nepal", "Rainas", "Sundarbazar", "Kwholasothar", "Dudhpokhari", "Dordi", "Marsyangdi"],
    "Mahottari": ["Jaleshwar", "Bardibas", "Gaushala", "Balwa", "Manara Shiswa", "Bhangaha", "Sonama", "Aurahi", "Ekdara", "Samsi", "Mahottari", "Pipra", "Matihani", "Ramgopalpur", "Loharpatti"],
    "Makwanpur": ["Hetauda", "Thaha", "Bakaiya", "Manahari", "Raksirang", "Makawanpurgadhi", "Kailash", "Bhimphedi", "Bagmati", "Indrasarowar"],
    "Manang": ["Chame", "Nason", "Narpa Bhumi", "Manang Ngisyang"],
    "Morang": ["Biratnagar", "Sundarharaicha", "Belbaari", "Pathari Shanischare", "Urlabari", "Letang", "Rangeli", "Sunawarshi", "Ratuwamai", "Kanepokhari"],
    "Mugu": ["Chhayanath Rara", "Mugum Karmarong", "Soru", "Khatyad"],
    "Mustang": ["Gharpajhong", "Thasang", "Barhagaun Muktichhetra", "Lo-Ghekar Damodarkunda", "Lomanthang"],
    "Myagdi": ["Beni", "Annapurna", "Dhaulagiri", "Mangala", "Malika", "Raghuganga"],
    "Nawalparasi East": ["Kawasoti", "Gaindakot", "Devchuli", "Madhyabindu", "Hupsekot", "Bulingtar", "Bungdikali", "Binayi Tribeni"],
    "Nawalparasi West": ["Ramgram", "Sunwal", "Bardaghat", "Sarawal", "Palhinandan", "Pratappur", "Susta"],
    "Nuwakot": ["Bidur", "Belkotgadhi", "Kakani", "Dupcheshwar", "Shivapuri", "Tadi", "Likhu", "Panchakanya", "Suryagadhi"],
    "Okhaldhunga": ["Siddhicharan", "Manebhanjyang", "Champadevi", "Sunkoshi", "Molung", "Chisankhugadhi", "Khijidemba", "Likhu"],
    "Palpa": ["Tansen", "Rampur", "Rainadevi Chhahara", "Tinau", "Mathagadhi", "Ribdikot", "Rambha", "Purbakhola", "Nisdi", "Bagnaskali"],
    "Panchthar": ["Phidim", "Phalelung", "Hilihang", "Phalgunanda", "Kummayak", "Miklajung", "Tumbewa", "Yangwarak"],
    "Parbat": ["Kusma", "Modi", "Jaljala", "Paiyun", "Phalewas", "Mahashila", "Bihadi"],
    "Parsa": ["Birgunj", "Pokhariya", "Bahudaramai", "Parsagadhi", "Bindabasini", "Dhobini", "Chhipaharmai", "Jagarnathpur", "Kalikamai", "Pakahamainpur"],
    "Pyuthan": ["Pyuthan", "Swargadwari", "Gaumukhi", "Mandavi", "Sarumarani", "Mallarani", "Naubahini", "Jhimruk", "Airawati"],
    "Ramechhap": ["Manthali", "Ramechhap", "Doramba", "Sunapati", "Khadadevi", "Gokulganga", "Likhu Tamakoshi", "Umakunda"],
    "Rasuwa": ["Gosaikunda", "Kalika", "Naukunda", "Uttargaya", "Amachodingmo"],
    "Rautahat": ["Chandrapur", "Garuda", "Gaur", "Baudhimai", "Brindaban", "Dewahi Gonahi", "Gadhimai", "Gujara", "Ishanath", "Katahariya", "Madhav Narayan", "Maulapur", "Paroha", "Phatuwa Bijayapur", "Rajdevi", "Rajpur", "Yamunamai"],
    "Rolpa": ["Rolpa", "Triveni", "Pariwartan", "Madi", "Runtigadhi", "Lungri", "Gangadev", "Sunchhahari", "Thabang", "Sunil Smriti"],
    "Rukum East": ["Sisne", "Bhume", "Putha Uttarganga"],
    "Rukum West": ["Musikot", "Chaurjahari", "Aathbiskot", "Banfikot", "Tribeni", "Sani Bheri"],
    "Rupandehi": ["Butwal", "Tilottama", "Siddharthanagar", "Lumbini Sanskritik", "Sainamaina", "Devdaha", "Mayadevi", "Rohini", "Marchawari", "Kotahimai"],
    "Salyan": ["Shaarda", "Bangad Kupinde", "Kalimati", "Tribeni", "Kapurkot", "Chatreshwori", "Siddha Kumakh", "Kumakh", "Dharma", "Baghchaur"],
    "Sankhuwasabha": ["Chainpur", "Khandbari", "Madi", "Panchkhapan", "Bhot Khola", "Chichila", "Makalu", "Sabhapokhari", "Silichong", "Dharmadevi"],
    "Saptari": ["Rajbiraj", "Kanchanrup", "Dakneshwori", "Bodebarsain", "Hanumannagar Kankalini", "Khadak", "Sambhunath", "Surunga", "Bishnupur", "Rupani", "Rajgadh", "Mahadeya", "Tirahut", "Tilathi Koiladi"],
    "Sarlahi": ["Malangwa", "Lalbandi", "Hariwan", "Ishworpur", "Barahathawa", "Balara", "Godaita", "Bagmati", "Kabilasi", "Chakraghatta", "Ramnagar", "Chandranagar", "Dhankaul", "Bramhapuri", "Bishnu", "Kaudena", "Parsa", "Basbariya", "Haripurwa"],
    "Sindhuli": ["Kamalamai", "Dudhauli", "Golanjor", "Ghyanglekh", "Tinpatan", "Marin", "Sunkoshi", "Hariharpurgadhi", "Phikkal"],
    "Sindhupalchok": ["Chautara Sangachokgadhi", "Balefi", "Melamchi", "Indrawati", "Jugal", "Panchpokhari Thangpal", "Sunkoshi", "Bhotekoshi", "Lisankhu Pakhar", "Barhabise", "Helambu", "Tripurasundari"],
    "Siraha": ["Lahan", "Dhangadhimai", "Siraha", "Golbazar", "Mirchaiya", "Kalyanpur", "Karjanha", "Sukhipur", "Bhagwanpur", "Aurahi"],
    "Solukhumbu": ["Solududhkunda", "Dudhakaushika", "Necha Salyan", "Dudhkoshi", "Maha Kulung", "Sotang", "Khumbu Pasanglhamu", "Thulung Dudhkoshi", "Likhu Pike"],
    "Sunsari": ["Itahari", "Dharan", "Inaruwa", "Duhabi", "Ramdhuni", "Barah", "Dewangunj", "Koshi", "Gadhi", "Barju", "Bhokraha Narsingh"]
    # ... add other districts and their cities here ...
}

# Function to get data from Google Places API (handles pagination)
def fetch_places(query, next_page_token=None):
    params = {
        'input': query, 
        'inputtype': 'textquery',
        'fields': 'formatted_address,name,rating,geometry',
        'key': API_KEY
    }

    if next_page_token:
        params['pagetoken'] = next_page_token  # Use the token to fetch the next set of results
    try:
        response = requests.get(PLACE_SEARCH_URL, params=params)
        response.raise_for_status()  # Ensure we raise an exception for HTTP errors
        return response.json()
    except requests.RequestException as e:
            print(f"Error fetching places for query '{query}': {e}")
            return {}

# Function to extract required fields, filtering based on types and ensuring district is in the address

def extract_tourist_places(data, district):
    """Extract relevant data from API response."""
    places = []
    candidates = data.get('candidates', [])
    print(f"Total candidates received: {len(candidates)}")
    for place in candidates:  # Use 'candidates' for Find Place API
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
            'types': ', '.join(place.get('types', [])),  # Store types as comma-separated string
        })
    print(f"Filtered places: {len(places)}")
    return places

# Function to save data to SQLite database after each district
def save_to_database(places):
    if not places:
        print("No places to save to the database.")
        return

    conn = sqlite3.connect('Dataset/tourist_attractions.db')
    cursor = conn.cursor()

    # Create attractions table with necessary columns (including the new 'types' and 'query_type' columns)
    cursor.execute(''' 
        CREATE TABLE IF NOT EXISTS attractions_nepal (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            address TEXT,
            latitude REAL,
            longitude REAL,
            rating REAL,
            second_word TEXT,
            types TEXT,  -- Column for types
            UNIQUE(name, latitude, longitude)  -- Ensure uniqueness based on name, latitude, and longitude
        )
    ''')

    # Insert places into the database (avoid duplicates)
    for place in places:
        cursor.execute(''' 
            SELECT id, types FROM attractions_nepal WHERE name = ? AND latitude = ? AND longitude = ?
        ''', (place['name'], place['latitude'], place['longitude']))
        existing_entry = cursor.fetchone()
        
        if existing_entry:
            print(f"Duplicate found for place: {place['name']} - {place['address']}")
            existing_id, existing_types = existing_entry
            if not existing_types or existing_types == 'N/A':
                print(f"Updating types for place: {place['name']} - {place['address']}")
                cursor.execute(''' 
                    UPDATE attractions_nepal SET types = ? WHERE id = ?
                ''', (place.get('types', 'N/A'), existing_id))
            continue

        # Ensure 'types' has a default value if missing and append query_type
        types = place.get('types', 'N/A')
        if types != 'N/A':
            types += f", {place['query_type']}"
        else:
            types = place['query_type']

        try:
            cursor.execute(''' 
                INSERT INTO attractions_nepal (name, address, latitude, longitude, rating, second_word, types)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (place['name'], place['address'], place['latitude'], place['longitude'], place['rating'], place['second_word'], types))
        except sqlite3.IntegrityError as e:
            print(f"Error inserting {place['name']} - {place['address']}: {e}")

    conn.commit()
    conn.close()

# Input parameters
types_keywords = ['park','amusement_park', 'church','tourist_attraction','mosque','museum', 'zoo', 'aquarium', 'art_gallery', 'hindu_temple', 'monument', 'historic_site', 'trek', 'picnic', 'hike']  # List of types/keywords to filter

# Function to handle paginated results
def get_all_places(query, district):
    places = []
    next_page_token = None
    while True:
        print(f"Performing query: {query}")
        response_data = fetch_places(query, next_page_token)
        print(f"Received response: {response_data}")
        new_places = extract_tourist_places(response_data, district)
        places.extend(new_places)
        next_page_token = response_data.get('next_page_token')
        
        if not next_page_token:
            break
        
        # Wait for the next page token to be available (must be at least 2 seconds)
        print(f"Waiting for next page in district: {district}...")
        time.sleep(1)
    return places

# Loop through all districts and their cities, saving after each district
for district, cities in districts_cities.items():
    all_places = []
    for city in cities:
        for kw in types_keywords:
            query_kw = f"{kw} in {city}, {district}, Nepal"
            places = get_all_places(query_kw, district)
            for place in places:
                place['query_type'] = kw  # Add query type to each place
            all_places.extend(places)
    print(f"Fetched {len(all_places)} total places for district: {district}")
    save_to_database(all_places)
    print(f"Saved tourist places for district: {district}")
    time.sleep(2)

print("Data collection and saving process is complete.")
