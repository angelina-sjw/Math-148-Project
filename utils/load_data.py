import os
import json
import sqlite3
import pandas as pd

from processing import process_parking_ambience_categories
from zipcode import merge_income_data


def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create the review, user, and business tables in the SQLite database if they do not exist.
    """
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS review (
        user_id TEXT, 
        business_id TEXT, 
        stars REAL, 
        useful INTEGER, 
        cool INTEGER, 
        text TEXT
    )""")
    
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
    
    conn.commit()


def load_filtered_json_to_sqlite(file_path: str, table_name: str, columns: list, conn: sqlite3.Connection) -> None:
    """
    Read a JSON file line by line, extract specified columns, and insert records into the SQLite database.
    For the 'business' table, additional attributes (e.g., parking and ambience) are extracted.
    Insertion is performed in batches of 10,000 records for efficiency.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        batch = []
        for line in f:
            record = json.loads(line)
            # Extract only the required columns
            selected_data = {col: record.get(col, None) for col in columns}

            # For the business table, extract additional attribute columns
            if table_name == "business":
                attributes = record.get("attributes", {}) or {}
                selected_data["BusinessParking"] = json.dumps(attributes.get("BusinessParking", {}))
                selected_data["Ambience"] = json.dumps(attributes.get("Ambience", {}))
                selected_data["RestaurantsAttire"] = attributes.get("RestaurantsAttire", None)
                selected_data["RestaurantsPriceRange2"] = attributes.get("RestaurantsPriceRange2", None)

            batch.append(tuple(selected_data.values()))
            # Insert in batches of 10,000 records
            if len(batch) >= 10000:
                placeholders = ', '.join(['?'] * len(selected_data))
                conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", batch)
                conn.commit()
                batch = []

        # Insert any remaining records
        if batch:
            placeholders = ', '.join(['?'] * len(selected_data))
            conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", batch)
            conn.commit()


def merge_and_process_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Merge data from the review, business, and user tables into a single DataFrame.
    Then, expand parking/ambience attributes and merge income data.
    """
    query = """
    SELECT r.user_id, r.business_id, r.stars, r.useful, r.cool, r.text,
           b.name, b.address, b.city, b.state, b.postal_code, b.latitude, b.longitude, 
           b.stars AS business_stars, b.review_count AS business_review_count, 
           b.BusinessParking, b.Ambience, b.RestaurantsAttire, b.RestaurantsPriceRange2, b.categories,
           u.review_count AS user_review_count, u.useful AS user_useful, u.funny AS user_funny, u.cool AS user_cool, 
           u.average_stars, u.fans, u.compliment_hot, u.compliment_more, u.compliment_profile, u.compliment_cute, 
           u.compliment_list, u.compliment_note, u.compliment_plain, u.compliment_cool, u.compliment_funny, 
           u.compliment_writer, u.compliment_photos
    FROM review r
    LEFT JOIN business b ON r.business_id = b.business_id
    LEFT JOIN user u ON r.user_id = u.user_id
    """
    df = pd.read_sql_query(query, conn)
    
    # Process parking, ambience categories and merge income data
    df = process_parking_ambience_categories(df)
    df = merge_income_data(df)
    return df


def filter_useful_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters and classifies reviews based on usefulness.
    
    Steps:
      1. Keep businesses with at least 10 reviews.
      2. Convert 'useful' to numeric and filter rows with useful > 0.
      3. Compute 40th and 60th percentiles of 'useful' per business.
      4. Classify reviews as 'less useful', 'average', or 'more useful' and filter out average.
    """
    # Filter for businesses with at least 10 reviews
    business_review_counts = df.groupby("business_id")["user_id"].count()
    valid_businesses = business_review_counts[business_review_counts >= 10].index
    df = df[df["business_id"].isin(valid_businesses)].copy()

    # Ensure 'useful' is numeric and filter reviews with useful > 0
    df["useful"] = pd.to_numeric(df["useful"], errors='coerce')
    df = df[df["useful"] > 0].copy()
    df.dropna(subset=['useful', 'business_id'], inplace=True)

    # Compute the 40th and 60th percentiles of 'useful' per business
    df['p40'] = df.groupby('business_id')['useful'].transform(lambda x: x.quantile(0.40))
    df['p60'] = df.groupby('business_id')['useful'].transform(lambda x: x.quantile(0.60))

    # Classify each review based on its 'useful' score relative to percentiles
    def classify_useful(row) -> str:
        if row['useful'] < row['p40']:
            return "less useful"
        elif row['useful'] > row['p60']:
            return "more useful"
        else:
            return "average"

    df['useful_category'] = df.apply(classify_useful, axis=1)
    # Exclude reviews classified as 'average'
    df = df[df['useful_category'] != 'average']
    # Remove temporary percentile columns
    df.drop(['p40', 'p60'], axis=1, inplace=True)
    
    return df


def main() -> None:
    """
    Main function to load JSON data into a SQLite database, merge tables, process data, and export the filtered dataset.
    """
    # Define file paths
    data_folder = "yelp_dataset"
    review_file = os.path.join(data_folder, 'yelp_academic_dataset_review.json')
    user_file = os.path.join(data_folder, 'yelp_academic_dataset_user.json')
    business_file = os.path.join(data_folder, 'yelp_academic_dataset_business.json')
    db_file = os.path.join(data_folder, 'yelp_data.db')

    # Connect to (or create) the SQLite database and create tables
    conn = sqlite3.connect(db_file)
    create_tables(conn)

    # Mapping: file -> (table name, columns to extract)
    tables_columns = {
        review_file: ("review", ["user_id", "business_id", "stars", "useful", "cool", "text"]),
        user_file: ("user", ["user_id", "review_count", "useful", "funny", "cool", "average_stars", "fans", 
                             "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", 
                             "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", 
                             "compliment_funny", "compliment_writer", "compliment_photos"]),
        business_file: ("business", ["business_id", "name", "address", "city", "state", "postal_code", 
                                     "latitude", "longitude", "stars", "review_count", "categories"])
    }

    # Load JSON files into corresponding SQL tables
    for file_path, (table, columns) in tables_columns.items():
        print(f"Loading {file_path} into table {table}...")
        load_filtered_json_to_sqlite(file_path, table, columns, conn)

    # Merge data from review, business, and user tables and process attributes/income data
    df_merged = merge_and_process_data(conn)
    # Filter and classify reviews based on usefulness
    df_filtered = filter_useful_reviews(df_merged)

    # Save the final DataFrame as a CSV file
    output_file = os.path.join(data_folder, 'yelp_merged_data_filtered.csv')
    df_filtered.to_csv(output_file, index=False)
    print(f"Merged dataset saved as {output_file}")

    conn.close()


if __name__ == '__main__':
    main()