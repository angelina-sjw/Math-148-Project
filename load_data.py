import os
import pandas as pd
import json
import sqlite3

# file paths
data_folder = "yelp_dataset"
review_file = os.path.join(data_folder, 'yelp_academic_dataset_review.json')
user_file = os.path.join(data_folder, 'yelp_academic_dataset_user.json')
business_file = os.path.join(data_folder, 'yelp_academic_dataset_business.json')
db_file = os.path.join(data_folder, 'yelp_data.db')

# connect to or create SQL database
conn = sqlite3.connect(db_file)
cursor = conn.cursor()

# review data
cursor.execute("""
CREATE TABLE IF NOT EXISTS review (
    user_id TEXT, 
    business_id TEXT, 
    stars REAL, 
    useful INTEGER, 
    cool INTEGER, 
    text TEXT
)""")

# user data
cursor.execute("""
CREATE TABLE IF NOT EXISTS user (
    user_id TEXT PRIMARY KEY, 
    review_count INTEGER, 
    useful INTEGER, 
    funny INTEGER, 
    cool INTEGER, 
    average_stars REAL, 
    fans INTEGER, 
    compliment_hot INTEGER, 
    compliment_more INTEGER, 
    compliment_profile INTEGER, 
    compliment_cute INTEGER, 
    compliment_list INTEGER, 
    compliment_note INTEGER, 
    compliment_plain INTEGER, 
    compliment_cool INTEGER, 
    compliment_funny INTEGER, 
    compliment_writer INTEGER, 
    compliment_photos INTEGER
)""")

# business data
cursor.execute("""
CREATE TABLE IF NOT EXISTS business (
    business_id TEXT PRIMARY KEY, 
    name TEXT, 
    address TEXT, 
    city TEXT, 
    state TEXT, 
    postal_code TEXT, 
    latitude REAL, 
    longitude REAL, 
    stars REAL, 
    review_count INTEGER, 
    categories TEXT, 
    BusinessParking TEXT, 
    Ambience TEXT, 
    RestaurantsAttire TEXT, 
    RestaurantsPriceRange2 TEXT
)""")

conn.commit() # commit schema changes


def load_filtered_json_to_sqlite(file_path, table_name, columns, conn):
    
    """
    Reads a JSON file line by line, extracts relevant columns, and inserts data into the SQLite database.
    For the 'business' table, extracts additional attributes (e.g., parking, ambience, attire, price range).
    Uses batch insertion to improve performance.
    """
    
    with open(file_path, 'r', encoding='utf-8') as f:
        batch = []
        for line in f:
            record = json.loads(line)
            selected_data = {col: record.get(col, None) for col in columns}
            
            if table_name == "business":
                attributes = record.get("attributes", {}) or {}
                selected_data["BusinessParking"] = json.dumps(attributes.get("BusinessParking", {}))
                selected_data["Ambience"] = json.dumps(attributes.get("Ambience", {}))
                selected_data["RestaurantsAttire"] = attributes.get("RestaurantsAttire", None)
                selected_data["RestaurantsPriceRange2"] = attributes.get("RestaurantsPriceRange2", None)
            
            batch.append(tuple(selected_data.values()))
            # batch insert every 10,000 records for efficiency
            if len(batch) >= 10000:
                placeholders = ', '.join(['?'] * len(selected_data))
                conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", batch)
                conn.commit()
                batch = []
        
        # insert remaining records
        if batch:
            placeholders = ', '.join(['?'] * len(selected_data))
            conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", batch)
            conn.commit()

# table mappings for data ingestion
tables_columns = {
    review_file: ("review", ["user_id", "business_id", "stars", "useful", "cool", "text"]),
    user_file: ("user", ["user_id", "review_count", "useful", "funny", "cool", "average_stars", "fans", "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", "compliment_funny", "compliment_writer", "compliment_photos"]),
    business_file: ("business", ["business_id", "name", "address", "city", "state", "postal_code", "latitude", "longitude", "stars", "review_count", "categories"])
}

# load each dataset json file into the corresponding SQL table
for file, (table, columns) in tables_columns.items():
    load_filtered_json_to_sqlite(file, table, columns, conn)

# join all data into a single dataset
query = """
SELECT r.user_id, r.business_id, r.stars, r.useful, r.cool, r.text,
       b.name, b.address, b.city, b.state, b.postal_code, b.latitude, b.longitude, b.stars AS business_stars, b.review_count AS business_review_count, 
       b.BusinessParking, b.Ambience, b.RestaurantsAttire, b.RestaurantsPriceRange2, b.categories,
       u.review_count AS user_review_count, u.useful AS user_useful, u.funny AS user_funny, u.cool AS user_cool, 
       u.average_stars, u.fans, u.compliment_hot, u.compliment_more, u.compliment_profile, u.compliment_cute, 
       u.compliment_list, u.compliment_note, u.compliment_plain, u.compliment_cool, u.compliment_funny, u.compliment_writer, u.compliment_photos
FROM review r
LEFT JOIN business b ON r.business_id = b.business_id
LEFT JOIN user u ON r.user_id = u.user_id
"""

# execute query and save to CSV
df_merged = pd.read_sql_query(query, conn)
df_merged.to_csv(os.path.join(data_folder, 'yelp_merged_data.csv'), index=False)

print("Merged dataset saved as yelp_merged_data.csv")

conn.close() # close connection