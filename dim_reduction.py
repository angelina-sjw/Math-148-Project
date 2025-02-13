# dim_reduction.py
import pandas as pd

def reduce_dimensions(input_csv, output_csv, chunksize=50000):
    """
    1) Keep only relevant columns for final ML usage.
    2) Merge/transform related columns:
       - user_funny, user_cool, user_useful => user_total_votes
       - compliment_* => user_total_compliments
       - parking booleans => parking_count
       - ambience booleans => ambience_count
       - Families Median/Mean => region_income
       - RestaurantsPriceRange2 => price_binary (cheap / expensive)
       - RestaurantsAttire => attire_binary (casual / formal)
    3) Drop the original columns after merging to reduce dimension.
    4) Keep some ID columns (user_id, business_id) and text if needed.
    5) Keep 'useful_category' as the target.
    """

    # Step A: define which columns to keep initially before merges
    # (Include everything needed for merging + final kept columns)
    keep_cols = [
        # IDs
        "user_id", "business_id",
        # review-level
        "text",
        "stars", "useful", "cool",  # optional: might cause leakage if 'useful_category' depends on these
        "useful_category",          # target label
        # user-level
        "user_review_count",
        "user_useful", "user_funny", "user_cool",  # to sum => user_total_votes
        "average_stars", "fans",
        # compliments
        "compliment_hot", "compliment_more", "compliment_profile",
        "compliment_cute", "compliment_list", "compliment_note",
        "compliment_plain", "compliment_cool", "compliment_funny",
        "compliment_writer", "compliment_photos",
        # business-level
        "business_stars", "business_review_count",
        # parking booleans
        "garage", "street", "validated", "lot", "valet",
        # ambience booleans
        "touristy", "hipster", "romantic", "divey", "intimate",
        "trendy", "upscale", "classy", "casual",
        # income
        "Families Median Income (Dollars)", "Families Mean Income (Dollars)",
        "zip_code",
        # transform category
        "RestaurantsPriceRange2", "RestaurantsAttire",
    ]
    # We intentionally do NOT include columns we want to drop
    # such as name, address, city, state, latitude, longitude,
    # BusinessParking, Ambience, etc.

    # Step B: define merges
    compliment_cols = [
        "compliment_hot", "compliment_more", "compliment_profile",
        "compliment_cute", "compliment_list", "compliment_note",
        "compliment_plain", "compliment_cool", "compliment_funny",
        "compliment_writer", "compliment_photos",
    ]
    parking_cols = ["garage", "street", "validated", "lot", "valet"]
    ambience_cols = ["touristy", "hipster", "romantic", "divey", "intimate",
                     "trendy", "upscale", "classy", "casual"]
    user_vote_cols = ["user_useful", "user_funny", "user_cool"]

    first_chunk = True

    for chunk in pd.read_csv(input_csv, chunksize=chunksize, low_memory=False):
        # 1) Filter columns to keep
        exist_cols = list(set(keep_cols).intersection(chunk.columns))
        chunk = chunk[exist_cols]

        # 2) Merge user votes => user_total_votes
        valid_votes = [c for c in user_vote_cols if c in chunk.columns]
        if valid_votes:
            chunk["user_total_votes"] = chunk[valid_votes].sum(axis=1, numeric_only=True)
        else:
            chunk["user_total_votes"] = 0

        # 3) Merge compliments => user_total_compliments
        valid_comps = [c for c in compliment_cols if c in chunk.columns]
        if valid_comps:
            chunk["user_total_compliments"] = chunk[valid_comps].sum(axis=1, numeric_only=True)
        else:
            chunk["user_total_compliments"] = 0

        # 4) Merge parking => parking_count
        valid_parking = [c for c in parking_cols if c in chunk.columns]
        if valid_parking:
            chunk["parking_count"] = chunk[valid_parking].sum(axis=1, numeric_only=True)
        else:
            chunk["parking_count"] = 0

        # 5) Merge ambience => ambience_count
        valid_amb = [c for c in ambience_cols if c in chunk.columns]
        if valid_amb:
            chunk["ambience_count"] = chunk[valid_amb].sum(axis=1, numeric_only=True)
        else:
            chunk["ambience_count"] = 0

        # 6) Merge incomes => region_income
        has_median = "Families Median Income (Dollars)" in chunk.columns
        has_mean   = "Families Mean Income (Dollars)" in chunk.columns
        if has_median and has_mean:
            chunk["region_income"] = (chunk["Families Median Income (Dollars)"] +
                                      chunk["Families Mean Income (Dollars)"]) / 2
        elif has_median:
            chunk["region_income"] = chunk["Families Median Income (Dollars)"]
        elif has_mean:
            chunk["region_income"] = chunk["Families Mean Income (Dollars)"]
        else:
            chunk["region_income"] = 0

        # 7) Convert RestaurantsPriceRange2 => price_binary
        if "RestaurantsPriceRange2" in chunk.columns:
            # e.g. 1 or 2 => cheap, 3 or 4 => expensive
            def price_map(x):
                if pd.isnull(x):
                    return None
                # Some Yelp data has them as int or string
                try:
                    val = int(x)
                    if val in [1,2]:
                        return "cheap"
                    elif val in [3,4]:
                        return "expensive"
                    else:
                        return None
                except:
                    return None
            chunk["price_binary"] = chunk["RestaurantsPriceRange2"].apply(price_map)

        # 8) Convert RestaurantsAttire => attire_binary (casual / formal)
        if "RestaurantsAttire" in chunk.columns:
            def attire_map(x):
                if isinstance(x, str):
                    # e.g. "casual", "dressy", "formal" in Yelp data
                    lowerx = x.lower()
                    if "casual" in lowerx:
                        return "casual"
                    else:
                        return "formal"
                return None
            chunk["attire_binary"] = chunk["RestaurantsAttire"].apply(attire_map)

        # 9) Now drop columns that are merged or not needed for final ML
        drop_cols = []
        # user votes
        drop_cols += valid_votes  # user_useful, user_funny, user_cool
        # compliments
        drop_cols += valid_comps
        # parking
        drop_cols += valid_parking
        # ambience
        drop_cols += valid_amb
        # incomes
        if has_median: drop_cols.append("Families Median Income (Dollars)")
        if has_mean:   drop_cols.append("Families Mean Income (Dollars)")
        # price range + attire
        drop_cols += ["RestaurantsPriceRange2", "RestaurantsAttire"]

        # Make sure we only drop what's actually in chunk
        drop_cols = [c for c in drop_cols if c in chunk.columns]
        chunk.drop(columns=drop_cols, inplace=True)

        # 10) Convert new numeric columns to numeric
        numeric_cols = [
            "stars", "useful", "cool", "user_review_count",
            "average_stars", "fans", "business_stars", "business_review_count",
            "user_total_votes", "user_total_compliments",
            "parking_count", "ambience_count", "region_income"
        ]
        for c in numeric_cols:
            if c in chunk.columns:
                chunk[c] = pd.to_numeric(chunk[c], errors="coerce")

        # Optional: drop rows with missing numeric data
        chunk.dropna(subset=numeric_cols, inplace=True)

        # 11) Save chunk
        mode = "w" if first_chunk else "a"
        chunk.to_csv(output_csv, index=False, mode=mode, header=first_chunk)
        first_chunk = False

if __name__ == "__main__":
    input_path = "/Users/kaitlyn/Documents/m148/Math-148-Project-main/yelp_dataset/yelp_merged_data_filtered.csv"
    output_path = "/Users/kaitlyn/Documents/m148/Math-148-Project-main/yelp_dataset/yelp_reduced.csv"

    reduce_dimensions(input_path, output_path)
    print("Dimension-reduced data saved to:", output_path)
