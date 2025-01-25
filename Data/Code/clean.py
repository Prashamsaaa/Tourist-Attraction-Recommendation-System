import pandas as pd

# Load the dataset from the provided manual.csv file
data = pd.read_csv('Data/FinalDataset/manual.csv')

# Verified locations and types for entries from paste.txt
verified_data = {
    105: {"location": "Lamachaur, Pokhara", "types": "Hindu Temple, Religious"},
    109: {"location": "Gaowainadapur, Morang", "types": "Museum"},
    111: {"location": "Pulchowk, Lalitpur", "types": "Buddhist Monastery, Archaeological Site"},
    114: {"location": "Daikataela, Solukhumbu", "types": "Historic Site, Cultural Spot"},
    116: {"location": "Katahariya, Rautahat", "types": "Mosque, Religious"},
    117: {"location": "Pyuthan", "types": "Historic Site"},
    119: {"location": "Birgunj, Parsa", "types": "Mosque, Religious"},
    120: {"location": "Lamakai Chauha, Kailali", "types": "Church, Religious"},
    122: {"location": "Lamjung", "types": "Park, Amusement Spot"},
    125: {"location": "Amargadhi, Dadeldhura", "types": "Park, Recreational Area"},
    131: {"location": "Ambote, Dhankuta", "types": "Church, Religious"},
    134: {"location": "Khairahani, Chitwan", "types": "Art Gallery, Cultural Hub"},
    136: {"location": "Rangagaelai, Morang", "types": "Church, Community Hub"},
    137: {"location": "Pachakhal, Kavrepalanchok", "types": "Park, Scenic Spot"},
    139: {"location": "Rukum East", "types": "Cultural Landmark"},
    140: {"location": "Jayaprithvi, Bajhang", "types": "Church, Religious Site"},
    142: {"location": "Kathmandu", "types": "Art Gallery, Museum"},
    143: {"location": "Gaur-10, Rautahat", "types": "Historic Palace, Cultural Landmark"},
    146: {"location": "Dhading Besi, Dhading", "types": "Tourist Attraction"},
    148: {"location": "Jautapani, Makwanpur", "types": "Zoo"}
}

# Update the dataset with verified information
for index, row in data.iterrows():
    if row['id'] in verified_data:
        data.at[index, 'location'] = verified_data[row['id']]['location']
        data.at[index, 'types'] = verified_data[row['id']]['types']

# Save the updated dataset to a new CSV file
updated_file_path = 'Data/FinalDataset/updated_manual.csv'
data.to_csv(updated_file_path, index=False)

# Confirm the file has been saved
print(f"Updated dataset saved to {updated_file_path}")
