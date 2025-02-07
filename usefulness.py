import pandas as pd

# Load the sampled Yelp data
file_path = "yelp_sampled_data.csv" 
df = pd.read_csv(file_path)

business_review_counts = df.groupby("business_id")["user_id"].count()

min_reviews_threshold = 3
valid_businesses = business_review_counts[business_review_counts >= min_reviews_threshold].index

df_filtered_businesses = df[df["business_id"].isin(valid_businesses)].copy()

df_filtered = df_filtered_businesses[df_filtered_businesses["useful"] > 0].copy()


output_path = "filtered_yelp_data.csv"  
df_filtered.to_csv(output_path, index=False)

print(f"Original dataset size: {df.shape[0]} reviews")
print(f"Number of restaurants with at least {min_reviews_threshold} reviews: {len(valid_businesses)}")
print(f"Final dataset size after filtering: {df_filtered.shape[0]} reviews")
print(f"Cleaned dataset saved to: {output_path}")

print(df_filtered.head())