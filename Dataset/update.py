import sqlite3

# Connect to the SQLite database
db_path = "Dataset/tourist_attractions.db"  # Replace with the path to your SQLite database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Discard existing IDs
cursor.execute("UPDATE attractions_nepal SET id = NULL")

# Fetch all rows to reorder IDs
cursor.execute("SELECT rowid FROM attractions_nepal")
rows = cursor.fetchall()

# Assign new IDs to all rows
for index, (rowid,) in enumerate(rows, start=1):
    cursor.execute("UPDATE attractions_nepal SET id = ? WHERE rowid = ?", (index, rowid))

# Commit the changes and close the connection
conn.commit()
conn.close()

print("IDs discarded and reordered successfully.")
