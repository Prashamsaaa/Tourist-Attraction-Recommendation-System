import sqlite3

conn = sqlite3.connect('tourist_attractions.db')
cursor = conn.cursor()

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
    cursor.execute(sql_query)

    conn.commit()
    print("Name replacement completed successfully.")

    cursor.execute("SELECT * FROM manual")
    rows = cursor.fetchall()

except sqlite3.OperationalError as e:
    print(f"Error during execution: {e}")

finally:
    conn.close()
