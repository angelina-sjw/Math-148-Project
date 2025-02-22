import pandas as pd
import numpy as np
import os
from typing import Optional

def reduce_dimensions(input_csv: str, output_csv: str, chunksize: int = 50000) -> None:
    """
    Process a large CSV file in chunks to reduce its dimensions by performing several operations:
    
    1. Retain only relevant columns.
    2. Merge/transform columns:
       - Sum user vote columns into 'user_total_votes'.
       - Sum compliment columns into 'user_total_compliments'.
       - Sum parking indicator columns into 'parking_count'.
       - Sum ambience indicator columns into 'ambience_count'.
       - Combine income columns into 'region_income'.
       - Map 'RestaurantsPriceRange2' into a binary price category ('cheap' or 'expensive').
       - Map 'RestaurantsAttire' into a binary attire category ('casual' or 'formal').
    3. Drop the original columns that were merged or transformed.
    4. Convert key numeric columns to numeric types and drop rows with missing numeric values.
    
    The processed data is written to a new CSV file.
    """
    # Define the list of columns to retain from the input dataset.
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
    
    # Define groups of columns for aggregation.
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
        Map RestaurantsPriceRange2 values to a price category.
        Returns "cheap" for values 1 or 2, "expensive" for 3 or 4, and None otherwise.
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
        Map RestaurantsAttire to an attire category.
        Returns "casual" if the text contains "casual", otherwise "formal". 
        Returns None if input is not a valid string.
        """
        if isinstance(x, str):
            return "casual" if "casual" in x.lower() else "formal"
        return None

    first_chunk = True
    # Process the CSV file in chunks to manage memory usage.
    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        # Retain only the columns that are both in keep_cols and the current chunk.
        available_cols = list(set(keep_cols).intersection(chunk.columns))
        chunk = chunk[available_cols]

        # Sum user vote columns into 'user_total_votes'.
        valid_votes = [col for col in user_vote_cols if col in chunk.columns]
        chunk["user_total_votes"] = chunk[valid_votes].sum(axis=1, numeric_only=True) if valid_votes else 0

        # Sum compliment columns into 'user_total_compliments'.
        valid_comps = [col for col in compliment_cols if col in chunk.columns]
        chunk["user_total_compliments"] = chunk[valid_comps].sum(axis=1, numeric_only=True) if valid_comps else 0

        # Sum parking indicator columns into 'parking_count'.
        valid_parking = [col for col in parking_cols if col in chunk.columns]
        chunk["parking_count"] = chunk[valid_parking].sum(axis=1, numeric_only=True) if valid_parking else 0

        # Sum ambience indicator columns into 'ambience_count'.
        valid_ambience = [col for col in ambience_cols if col in chunk.columns]
        chunk["ambience_count"] = chunk[valid_ambience].sum(axis=1, numeric_only=True) if valid_ambience else 0

        # Combine income columns into 'region_income'.
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

        # Map 'RestaurantsPriceRange2' to a binary price category, if available.
        if "RestaurantsPriceRange2" in chunk.columns:
            chunk["price_binary"] = chunk["RestaurantsPriceRange2"].apply(map_price)

        # Map 'RestaurantsAttire' to a binary attire category, if available.
        if "RestaurantsAttire" in chunk.columns:
            chunk["attire_binary"] = chunk["RestaurantsAttire"].apply(map_attire)

        # Determine and drop the original columns that have been merged or transformed.
        drop_cols = valid_votes + valid_comps + valid_parking + valid_ambience
        if has_median:
            drop_cols.append("Families Median Income (Dollars)")
        if has_mean:
            drop_cols.append("Families Mean Income (Dollars)")
        drop_cols += ["RestaurantsPriceRange2", "RestaurantsAttire"]
        drop_cols = [col for col in drop_cols if col in chunk.columns]
        chunk.drop(columns=drop_cols, inplace=True)

        # Convert key columns to numeric types and drop rows with missing numeric values.
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

        # Write the processed chunk to the output CSV file.
        mode = "w" if first_chunk else "a"
        header = first_chunk
        chunk.to_csv(output_csv, index=False, mode=mode, header=header)
        first_chunk = False

if __name__ == "__main__":
    INPUT_PATH = "yelp_dataset/yelp_merged_data_filtered.csv"
    OUTPUT_PATH = "yelp_dataset/yelp_reduced.csv"
    reduce_dimensions(INPUT_PATH, OUTPUT_PATH)
    print("Dimension-reduced data saved to:", OUTPUT_PATH)
