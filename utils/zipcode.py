import pandas as pd
import os
import kagglehub

def merge_income_data(df, target_year = 2021):
    """
    Merge the input DataFrame with US household income data by ZIP code,
    filtering the income data for the specified target year (default is 2021).
    """
    # download the income dataset from Kaggle
    dataset_path = kagglehub.dataset_download("claygendron/us-household-income-by-zip-code-2021-2011")
    csv_file = os.path.join(dataset_path, "us_income_zipcode.csv")
    zip_income = pd.read_csv(csv_file)
    
    # === info for the original Kaggle dataset ===
    print("=== Kaggle Income Dataset Insights ===")
    print("Initial Kaggle dataset shape:", zip_income.shape)
    print("Kaggle dataset columns:", zip_income.columns.tolist())
    print("First 5 rows:")
    print(zip_income.head(), "\n")
    
    # === filter for the target year (e.g., 2021) ===
    if 'Year' in zip_income.columns:
        zip_income = zip_income[zip_income['Year'] == target_year]
        print(f"Shape after filtering for year {target_year}:", zip_income.shape, "\n")
    else:
        print("Warning: 'Year' column not found in Kaggle dataset. Proceeding without year filtering.\n")
    
    # === extract and clean the relevant columns ===
    # We only need median income, mean income, and ZIP for merging.
    zip_combine = zip_income[['Families Median Income (Dollars)', 
                              'Families Mean Income (Dollars)', 
                              'ZIP']]
    print("Extracted columns shape:", zip_combine.shape)
    print("Number of unique ZIP codes before cleaning:", zip_combine['ZIP'].nunique(), "\n")
    
    zip_combine['ZIP'] = zip_combine['ZIP'].astype(str).str.zfill(5)
    # Check for missing values
    missing_values = zip_combine.isna().sum()
    print("Missing values in each column before cleaning:")
    print(missing_values, "\n")
    
    zip_cleaned = zip_combine.dropna()
    print("Shape after dropping rows with missing values:", zip_cleaned.shape)
    
    duplicate_zip_count = zip_cleaned.duplicated(subset='ZIP').sum()
    print("Number of duplicate ZIP codes after cleaning:", duplicate_zip_count)
    
    zip_cleaned = zip_cleaned.drop_duplicates(subset='ZIP')
    print("Shape after dropping duplicate ZIP codes:", zip_cleaned.shape, "\n")
    
    # === prepare the input DataFrame ===
    # Make sure that the postal_code column is formatted as a five-digit string
    df.loc[:, 'postal_code'] = df['postal_code'].astype(str).str.zfill(5)
    
    # === merge the DataFrames ===
    merged_df = df.merge(zip_cleaned, left_on='postal_code', right_on='ZIP', how='inner')
    
    # clean up the merged DataFrame: drop 'postal_code' and rename 'ZIP' to 'zip_code'
    merged_df = merged_df.drop(columns=['postal_code']).rename(columns={'ZIP': 'zip_code'})
    print("Merged DataFrame shape:", merged_df.shape)
    
    return merged_df
