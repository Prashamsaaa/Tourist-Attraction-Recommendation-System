import sqlite3

def create_new_table_from_existing(db_path, source_table, new_table, columns_to_copy):
    """
    Create a new table in the same SQLite database by selecting specific columns from an existing table.

    Args:
        db_path (str): Path to the SQLite database.
        source_table (str): Name of the source table to copy data from.
        new_table (str): Name of the new table to create.
        columns_to_copy (list): List of column names to copy into the new table.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the source table exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{source_table}';")
    if not cursor.fetchone():
        print(f"Source table '{source_table}' does not exist.")
        conn.close()
        return
    
    # Check if the new table already exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{new_table}';")
    if cursor.fetchone():
        print(f"New table '{new_table}' already exists.")
        conn.close()
        return
    
    # Get the list of columns in the source table
    cursor.execute(f"PRAGMA table_info({source_table});")
    existing_columns = [info[1] for info in cursor.fetchall()]
    
    # Ensure the requested columns exist in the source table
    for column in columns_to_copy:
        if column not in existing_columns:
            print(f"Column '{column}' does not exist in the source table '{source_table}'.")
            conn.close()
            return
    
    # Create the new table by copying data from the source table
    columns_str = ", ".join(columns_to_copy)
    cursor.execute(f"CREATE TABLE {new_table} AS SELECT {columns_str} FROM {source_table};")
    conn.commit()
    conn.close()
    print(f"New table '{new_table}' created with columns {columns_to_copy} from '{source_table}'.")


def create_filtered_table_by_country(db_path, source_table, new_table, columns_to_copy, country_name):
    """
    Create a new table in the same SQLite database by selecting specific columns from an existing table
    and filtering rows based on the country name.

    Args:
        db_path (str): Path to the SQLite database.
        source_table (str): Name of the source table to copy data from.
        new_table (str): Name of the new table to create.
        columns_to_copy (list): List of column names to copy into the new table.
        country_name (str): Name of the country to filter rows by.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if the source table exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{source_table}';")
    if not cursor.fetchone():
        print(f"Source table '{source_table}' does not exist.")
        conn.close()
        return
    
    # Check if the new table already exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{new_table}';")
    if cursor.fetchone():
        print(f"New table '{new_table}' already exists.")
        conn.close()
        return
    
    # Get the list of columns in the source table
    cursor.execute(f"PRAGMA table_info({source_table});")
    existing_columns = [info[1] for info in cursor.fetchall()]
    
    # Ensure the requested columns exist in the source table
    for column in columns_to_copy:
        if column not in existing_columns:
            print(f"Column '{column}' does not exist in the source table '{source_table}'.")
            conn.close()
            return
    
    # Create the new table by copying data from the source table and filtering by country
    columns_str = ", ".join(columns_to_copy)
    cursor.execute(f"CREATE TABLE {new_table} AS SELECT {columns_str} FROM {source_table} WHERE Country='{country_name}';")
    conn.commit()
    conn.close()
    print(f"New table '{new_table}' created with columns {columns_to_copy} from '{source_table}' filtered by country '{country_name}'.")


def remove_duplicates(db_path, table_name):
    """
    Remove duplicate rows from the specified table in an SQLite database.
    
    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table from which to remove duplicates.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get the list of columns in the table
    cursor.execute(f"PRAGMA table_info({table_name});")
    columns = [info[1] for info in cursor.fetchall()]
    
    # Create a new table with the same structure as the original table
    columns_str = ", ".join(columns)
    new_table_name = f"{table_name}_no_duplicates"
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {new_table_name} AS
        SELECT DISTINCT {columns_str}
        FROM {table_name};
    """)
    
    # Drop the original table and rename the new table to the original table's name
    cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
    cursor.execute(f"ALTER TABLE {new_table_name} RENAME TO {table_name};")
    
    conn.commit()
    conn.close()
    
    print(f"Duplicates removed from table '{table_name}'.")


def move_unknown_cities_left(db_path, source_table="First_Phase_Done1", new_table="First_Phase_Left"):
    """
    Move rows with 'Unknown' city from the source table to a new table.
    
    Args:
        db_path (str): Path to the SQLite database.
        source_table (str): Name of the source table to copy data from.
        new_table (str): Name of the new table to create and move rows to.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the new table if it doesn't already exist
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {new_table} (
            ID INTEGER,
            Name TEXT,
            Latitude REAL,
            Longitude REAL,
            Country TEXT,
            City TEXT
        );
    """)
    
    # Move rows with 'Unknown' city to the new table
    cursor.execute(f"""
        INSERT INTO {new_table} (ID, Name, Latitude, Longitude, Country, City)
        SELECT ID, Name, Latitude, Longitude, Country, City
        FROM {source_table}
        WHERE City = 'Unknown';
    """)
    conn.commit()
    conn.close()
    print(f"Unknown city rows moved from {source_table} to {new_table}.")
    
def delete_unknown_city_rows(db_path, table_name):
    """
    Delete rows from the specified table where the city is 'Unknown'.

    Args:
        db_path (str): Path to the SQLite database.
        table_name (str): Name of the table from which to delete rows with 'Unknown' city.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Delete rows with 'Unknown' city
    cursor.execute(f"DELETE FROM {table_name} WHERE City = 'Unknown';")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print(f"Rows with 'Unknown' city deleted from table '{table_name}'.")

def move_table_content(db_path, source_table, target_table):
    """
    Move content from the source table to the target table.

    Args:
        db_path (str): Path to the SQLite database.
        source_table (str): Name of the source table.
        target_table (str): Name of the target table.
    """
    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if the source table exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{source_table}';")
    if not cursor.fetchone():
        print(f"Source table '{source_table}' does not exist.")
        conn.close()
        return

    # Check if the target table exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{target_table}';")
    if not cursor.fetchone():
        print(f"Target table '{target_table}' does not exist.")
        conn.close()
        return

    # Move content from source table to target table
    cursor.execute(f"INSERT INTO {target_table} SELECT * FROM {source_table};")

    # Commit the changes and close the connection
    conn.commit()
    conn.close()

    print(f"Content from table '{source_table}' moved to table '{target_table}'.")



# Example usage
if __name__ == "__main__":
    db_path = "Dataset/tourist_attractions.db"  # Path to your SQLite database
    table_name = "attractions_nepal"     # Name of the table to remove duplicates from
    
    # Uncomment the function you want to use:

    # Remove duplicates from the table
    remove_duplicates(db_path, table_name)

    # Move rows with 'Unknown' city to a new table
    # move_unknown_cities_left(db_path)

    # Example for creating a new table from an existing one
    # source_table = "First_Phase"  # Name of the source table
    # columns_to_copy = ["ID", "Name", "City", "Country", "Latitude", "Longitude"]  # Columns to copy
    # new_table = "New_Table"  # Name of the new table
    # create_new_table_from_existing(db_path, source_table, new_table, columns_to_copy)

    # Example for creating a filtered table based on country
    # country_name = "Brunei"
    # new_filtered_table = "Brunei_Attractions"
    # create_filtered_table_by_country(db_path, "First_Phase", new_filtered_table, columns_to_copy, country_name)
    
    # Delete rows with 'Unknown' city from the table
    # delete_unknown_city_rows(db_path, table_name)
    
    # source_table = "First_Phase_Done1"  # Name of the source table
    # target_table = "First_Phase_Done"  # Name of the target table

    # # Move content from source table to target table
    # move_table_content(db_path, source_table, target_table)
