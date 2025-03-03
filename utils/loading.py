import os
import json
import sqlite3
import pandas as pd
import numpy as np
import ast
from typing import Any, List, Optional
import kagglehub

# -----------------------
# Processing Functions
# -----------------------

def parse_dict_col(column: Any) -> dict:
    """
    Parse a string representation of a dictionary into an actual dictionary.
    """
    if isinstance(column, dict):
        return column
    try:
        # Remove extra quotes and evaluate the string safely.
        parsed_dict = ast.literal_eval(column.strip('"'))
        return parsed_dict if isinstance(parsed_dict, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def expand_dict_col(df: pd.DataFrame, column_name: str, expected_keys: List[str]) -> pd.DataFrame:
    """
    Expand a dictionary-like column into separate boolean (0/1) columns based on expected keys.
    """
    parsed_series = df[column_name].apply(parse_dict_col)
    expanded_df = pd.json_normalize(parsed_series)
    expanded_df = expanded_df.reindex(columns=expected_keys, fill_value=False)
    expanded_df = expanded_df.fillna(False).astype(int)
    return df.join(expanded_df)


def process_parking_ambience_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand 'BusinessParking' and 'Ambience' columns into separate boolean columns and drop 'categories'.
    """
    parking_cols = ["garage", "street", "validated", "lot", "valet"]
    ambience_cols = ["touristy", "hipster", "romantic", "divey", "intimate", "trendy", "upscale", "classy", "casual"]
    df = expand_dict_col(df, "BusinessParking", parking_cols)
    df = expand_dict_col(df, "Ambience", ambience_cols)
    # Remove the original categories column
    return df.drop(columns=["categories"])


def merge_income_data(df: pd.DataFrame, target_year: int = 2021) -> pd.DataFrame:
    """
    Merge the DataFrame with US household income data by ZIP code.
    """
    # Download and read the Kaggle income dataset.
    dataset_path: str = kagglehub.dataset_download("claygendron/us-household-income-by-zip-code-2021-2011")
    csv_file: str = os.path.join(dataset_path, "us_income_zipcode.csv")
    zip_income: pd.DataFrame = pd.read_csv(csv_file)

    print("=== Kaggle Income Dataset Insights ===")
    print("Initial Kaggle dataset shape:", zip_income.shape)
    print("Kaggle dataset columns:", zip_income.columns.tolist())
    print("First 5 rows:")
    print(zip_income.head(), "\n")
    
    # Filter dataset by target year if available.
    if 'Year' in zip_income.columns:
        zip_income = zip_income[zip_income['Year'] == target_year]
        print(f"Shape after filtering for year {target_year}:", zip_income.shape, "\n")
    else:
        print("Warning: 'Year' column not found. Proceeding without year filtering.\n")
    
    # Select relevant income columns and clean ZIP codes.
    zip_combine: pd.DataFrame = zip_income[['Families Median Income (Dollars)', 
                                             'Families Mean Income (Dollars)', 'ZIP']]
    print("Extracted columns shape:", zip_combine.shape)
    print("Unique ZIP codes before cleaning:", zip_combine['ZIP'].nunique(), "\n")
    
    zip_combine['ZIP'] = zip_combine['ZIP'].astype(str).str.zfill(5)
    print("Missing values before cleaning:")
    print(zip_combine.isna().sum(), "\n")
    
    zip_cleaned: pd.DataFrame = zip_combine.dropna()
    print("Shape after dropping missing rows:", zip_cleaned.shape)
    
    duplicate_zip_count: int = zip_cleaned.duplicated(subset='ZIP').sum()
    print("Duplicate ZIP codes after cleaning:", duplicate_zip_count)
    zip_cleaned = zip_cleaned.drop_duplicates(subset='ZIP')
    print("Shape after dropping duplicates:", zip_cleaned.shape, "\n")
    
    # Clean and merge with the main DataFrame.
    df['postal_code'] = df['postal_code'].astype(str).str.zfill(5)
    merged_df: pd.DataFrame = df.merge(zip_cleaned, left_on='postal_code', right_on='ZIP', how='inner')
    merged_df = merged_df.drop(columns=['postal_code']).rename(columns={'ZIP': 'zip_code'})
    print("Merged DataFrame shape:", merged_df.shape)
    
    return merged_df

# -----------------------
# Database and CSV Pipeline Functions
# -----------------------

def create_tables(conn: sqlite3.Connection) -> None:
    """
    Create review, user, and business tables in the SQLite database if they do not exist.
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
    Load JSON lines from a file and insert selected columns into a SQLite table.
    For the 'business' table, additional attributes are extracted.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        batch = []
        for line in f:
            record = json.loads(line)
            selected_data = {col: record.get(col, None) for col in columns}
            # Process additional fields for business records.
            if table_name == "business":
                attributes = record.get("attributes", {}) or {}
                selected_data["BusinessParking"] = json.dumps(attributes.get("BusinessParking", {}))
                selected_data["Ambience"] = json.dumps(attributes.get("Ambience", {}))
                selected_data["RestaurantsAttire"] = attributes.get("RestaurantsAttire", None)
                selected_data["RestaurantsPriceRange2"] = attributes.get("RestaurantsPriceRange2", None)
            batch.append(tuple(selected_data.values()))
            if len(batch) >= 10000:
                placeholders = ', '.join(['?'] * len(selected_data))
                conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", batch)
                conn.commit()
                batch = []
        if batch:
            placeholders = ', '.join(['?'] * len(selected_data))
            conn.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", batch)
            conn.commit()


def merge_and_process_data(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Merge data from review, business, and user tables, expand attributes, and merge income data.
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
    df = process_parking_ambience_categories(df)
    df = merge_income_data(df)
    return df


def filter_useful_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter reviews based on business review count and usefulness, classifying them as less or more useful.
    """
    # Only include businesses with at least 10 reviews.
    business_review_counts = df.groupby("business_id")["user_id"].count()
    valid_businesses = business_review_counts[business_review_counts >= 10].index
    df = df[df["business_id"].isin(valid_businesses)].copy()
    
    # Convert usefulness to numeric and filter for positive values.
    df["useful"] = pd.to_numeric(df["useful"], errors='coerce')
    df = df[df["useful"] > 0].copy()
    df.dropna(subset=['useful', 'business_id'], inplace=True)
    
    # Compute the 40th and 60th percentiles for each business.
    df['p40'] = df.groupby('business_id')['useful'].transform(lambda x: x.quantile(0.40))
    df['p60'] = df.groupby('business_id')['useful'].transform(lambda x: x.quantile(0.60))
    
    # Classify reviews based on usefulness percentiles.
    def classify_useful(row) -> str:
        if row['useful'] < row['p40']:
            return "less useful"
        elif row['useful'] > row['p60']:
            return "more useful"
        else:
            return "average"
    
    df['useful_category'] = df.apply(classify_useful, axis=1)
    df = df[df['useful_category'] != 'average']
    df.drop(['p40', 'p60'], axis=1, inplace=True)
    return df

# -----------------------
# Dimension Reduction Functions
# -----------------------

def reduce_dimensions(input_csv: str, output_csv: str, chunksize: int = 50000) -> None:
    """
    Reduce the dimensions of a large CSV by merging and transforming columns, then write the result to a new CSV.
    """
    # Columns to retain in the final output.
    keep_cols = [
        "user_id", "business_id", "text", "stars", "useful", "cool", "useful_category",
        "user_review_count", "user_useful", "user_funny", "user_cool",
        "average_stars", "fans",
        "compliment_hot", "compliment_more", "compliment_profile",
        "compliment_cute", "compliment_list", "compliment_note",
        "compliment_plain", "compliment_cool", "compliment_funny",
        "compliment_writer", "compliment_photos",
        "business_stars", "business_review_count",
        "garage", "street", "validated", "lot", "valet",
        "touristy", "hipster", "romantic", "divey", "intimate",
        "trendy", "upscale", "classy", "casual",
        "Families Median Income (Dollars)", "Families Mean Income (Dollars)", "zip_code",
        "RestaurantsPriceRange2", "RestaurantsAttire"
    ]
    compliment_cols = [
        "compliment_hot", "compliment_more", "compliment_profile",
        "compliment_cute", "compliment_list", "compliment_note",
        "compliment_plain", "compliment_cool", "compliment_funny",
        "compliment_writer", "compliment_photos"
    ]
    parking_cols = ["garage", "street", "validated", "lot", "valet"]
    ambience_cols = ["touristy", "hipster", "romantic", "divey", "intimate", "trendy", "upscale", "classy", "casual"]
    user_vote_cols = ["user_useful", "user_funny", "user_cool"]

    def map_price(x: Optional[str]) -> Optional[str]:
        """
        Map price range to 'cheap' (1 or 2) or 'expensive' (3 or 4).
        """
        if pd.isnull(x):
            return None
        try:
            val = int(x)
            if val in [1, 2]:
                return "cheap"
            elif val in [3, 4]:
                return "expensive"
        except Exception:
            pass
        return None

    def map_attire(x: Optional[str]) -> Optional[str]:
        """
        Map restaurant attire to 'casual' if the word is present, otherwise 'formal'.
        """
        if isinstance(x, str):
            return "casual" if "casual" in x.lower() else "formal"
        return None

    first_chunk = True
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        # Keep only available columns from the predefined list.
        available_cols = [col for col in keep_cols if col in chunk.columns]
        chunk = chunk[available_cols]
        
        # Create new features by summing existing columns.
        valid_votes = [col for col in user_vote_cols if col in chunk.columns]
        chunk["user_total_votes"] = chunk[valid_votes].sum(axis=1, numeric_only=True) if valid_votes else 0
        
        valid_comps = [col for col in compliment_cols if col in chunk.columns]
        chunk["user_total_compliments"] = chunk[valid_comps].sum(axis=1, numeric_only=True) if valid_comps else 0
        
        valid_parking = [col for col in parking_cols if col in chunk.columns]
        chunk["parking_count"] = chunk[valid_parking].sum(axis=1, numeric_only=True) if valid_parking else 0
        
        valid_ambience = [col for col in ambience_cols if col in chunk.columns]
        chunk["ambience_count"] = chunk[valid_ambience].sum(axis=1, numeric_only=True) if valid_ambience else 0

        # Calculate region income by averaging median and mean incomes if both exist.
        has_median = "Families Median Income (Dollars)" in chunk.columns
        has_mean = "Families Mean Income (Dollars)" in chunk.columns
        if has_median and has_mean:
            chunk["region_income"] = (chunk["Families Median Income (Dollars)"] +
                                      chunk["Families Mean Income (Dollars)"]) / 2
        elif has_median:
            chunk["region_income"] = chunk["Families Median Income (Dollars)"]
        elif has_mean:
            chunk["region_income"] = chunk["Families Mean Income (Dollars)"]
        else:
            chunk["region_income"] = 0

        # Map price and attire fields to binary categories.
        if "RestaurantsPriceRange2" in chunk.columns:
            chunk["price_binary"] = chunk["RestaurantsPriceRange2"].apply(map_price)
        if "RestaurantsAttire" in chunk.columns:
            chunk["attire_binary"] = chunk["RestaurantsAttire"].apply(map_attire)

        # Drop original columns that have been merged into new features.
        drop_cols = valid_votes + valid_comps + valid_parking + valid_ambience
        if has_median:
            drop_cols.append("Families Median Income (Dollars)")
        if has_mean:
            drop_cols.append("Families Mean Income (Dollars)")
        drop_cols += ["RestaurantsPriceRange2", "RestaurantsAttire"]
        drop_cols = [col for col in drop_cols if col in chunk.columns]
        chunk.drop(columns=drop_cols, inplace=True)

        # Convert key columns to numeric types and drop rows with missing values.
        numeric_cols = [
            "stars", "useful", "cool", "user_review_count",
            "average_stars", "fans", "business_stars", "business_review_count",
            "user_total_votes", "user_total_compliments",
            "parking_count", "ambience_count", "region_income"
        ]
        for col in numeric_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk.dropna(subset=numeric_cols, inplace=True)

        # Write the processed chunk to the output CSV.
        mode = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(output_csv, index=False, mode=mode, header=header)
        first_chunk = False

# -----------------------
# Combined Main Function
# -----------------------

def main() -> None:
    """
    Execute the data processing pipeline: load JSON files into SQLite, merge and filter data,
    and reduce dimensions for the final output.
    """
    data_folder = "yelp_dataset"
    review_file = os.path.join(data_folder, 'yelp_academic_dataset_review.json')
    user_file = os.path.join(data_folder, 'yelp_academic_dataset_user.json')
    business_file = os.path.join(data_folder, 'yelp_academic_dataset_business.json')
    db_file = os.path.join(data_folder, 'yelp_data.db')

    # Load JSON data into SQLite.
    conn = sqlite3.connect(db_file)
    create_tables(conn)
    tables_columns = {
        review_file: ("review", ["user_id", "business_id", "stars", "useful", "cool", "text"]),
        user_file: ("user", ["user_id", "review_count", "useful", "funny", "cool", "average_stars", "fans", 
                             "compliment_hot", "compliment_more", "compliment_profile", "compliment_cute", 
                             "compliment_list", "compliment_note", "compliment_plain", "compliment_cool", 
                             "compliment_funny", "compliment_writer", "compliment_photos"]),
        business_file: ("business", ["business_id", "name", "address", "city", "state", "postal_code", 
                                     "latitude", "longitude", "stars", "review_count", "categories"])
    }
    for file_path, (table, columns) in tables_columns.items():
        print(f"Loading {file_path} into table {table}...")
        load_filtered_json_to_sqlite(file_path, table, columns, conn)
    
    # Merge, process, and filter the data.
    df_merged = merge_and_process_data(conn)
    df_filtered = filter_useful_reviews(df_merged)
    
    data_folder = "yelp_dataset"
    
    # Save the intermediate merged dataset.
    intermediate_file = os.path.join(data_folder, 'yelp_merged_data_filtered.csv')
    df_filtered.to_csv(intermediate_file, index=False)
    print(f"Intermediate merged dataset saved as {intermediate_file}")
    conn.close()
    
    # Reduce dimensions and save the final output.
    output_file = os.path.join(data_folder, 'yelp_reduced.csv')
    reduce_dimensions(intermediate_file, output_file)
    print(f"Dimension-reduced data saved as: {output_file}")

if __name__ == '__main__':
    main()