import sqlite3

# Connect to the database (replace 'your_database.db' with your actual database file)
conn = sqlite3.connect('tourist_attractions.db')
cursor = conn.cursor()

# Step 1: Create tables if they don't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS attractions_nepal (
    id INTEGER PRIMARY KEY,
    name TEXT,
    address TEXT,
    latitude REAL,
    longitude REAL,
    rating REAL,
    second_word TEXT,
    types TEXT
);
''')

cursor.execute('''
CREATE TABLE IF NOT EXISTS manual (
    id INTEGER PRIMARY KEY,
    name TEXT,
    latitude REAL,
    longitude REAL
);
''')

# Step 2: Execute the update query to replace names in the manual table
sql_query = """
UPDATE manual
SET name = (
    SELECT attractions_nepal.name
    FROM attractions_nepal
    WHERE manual.latitude = attractions_nepal.latitude
      AND manual.longitude = attractions_nepal.longitude
)
WHERE EXISTS (
    SELECT 1
    FROM attractions_nepal
    WHERE manual.latitude = attractions_nepal.latitude
      AND manual.longitude = attractions_nepal.longitude
);
"""

try:
    # Execute the query to update names
    cursor.execute(sql_query)

    # Commit changes
    conn.commit()
    print("Name replacement completed successfully.")

    # Fetch and display updated data from the manual table
    cursor.execute("SELECT * FROM manual")
    rows = cursor.fetchall()

except sqlite3.OperationalError as e:
    print(f"Error during execution: {e}")

finally:
    conn.close()
