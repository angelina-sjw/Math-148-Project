import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the processed dataset
file_path = "/Users/kaitlyn/Documents/m148/Math-148-Project-main/yelp_dataset/yelp_reduced.csv"
df = pd.read_csv(file_path)

# Select only numeric columns for correlation analysis
numeric_cols = [
    "stars", "useful", "cool", "user_review_count",
    "average_stars", "fans", "business_stars", "business_review_count",
    "user_total_votes", "user_total_compliments",
    "parking_count", "ambience_count", "region_income"
]
df_numeric = df[numeric_cols]

# Compute the correlation matrix
corr_matrix = df_numeric.corr()

# Plot the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Final Numeric Features")
plt.show()
