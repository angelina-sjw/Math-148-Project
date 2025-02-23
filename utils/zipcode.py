import os
import pandas as pd
import kagglehub
from typing import Any

def merge_income_data(df: pd.DataFrame, target_year: int = 2021) -> pd.DataFrame:
    """
    Merge the input DataFrame with US household income data by ZIP code.
    
    The function downloads the income dataset from Kaggle, filters the data for the 
    specified target year (default is 2021), cleans the ZIP code data, and merges 
    the income data with the input DataFrame based on ZIP codes.
    """
    # Download the income dataset from Kaggle
    dataset_path: str = kagglehub.dataset_download("claygendron/us-household-income-by-zip-code-2021-2011")
    csv_file: str = os.path.join(dataset_path, "us_income_zipcode.csv")
    zip_income: pd.DataFrame = pd.read_csv(csv_file)
    
    # Display basic insights about the downloaded dataset
    print("=== Kaggle Income Dataset Insights ===")
    print("Initial Kaggle dataset shape:", zip_income.shape)
    print("Kaggle dataset columns:", zip_income.columns.tolist())
    print("First 5 rows:")
    print(zip_income.head(), "\n")
    
    # Filter the dataset by the target year if the 'Year' column exists
    if 'Year' in zip_income.columns:
        zip_income = zip_income[zip_income['Year'] == target_year]
        print(f"Shape after filtering for year {target_year}:", zip_income.shape, "\n")
    else:
        print("Warning: 'Year' column not found. Proceeding without year filtering.\n")
    
    # Extract only the relevant income and ZIP code columns
    zip_combine: pd.DataFrame = zip_income[['Families Median Income (Dollars)', 
                                             'Families Mean Income (Dollars)', 'ZIP']]
    print("Extracted columns shape:", zip_combine.shape)
    print("Unique ZIP codes before cleaning:", zip_combine['ZIP'].nunique(), "\n")
    
    # Standardize ZIP codes: convert to string and ensure 5-digit format
    zip_combine['ZIP'] = zip_combine['ZIP'].astype(str).str.zfill(5)
    
    # Check and display missing values before cleaning
    missing_values: pd.Series = zip_combine.isna().sum()
    print("Missing values before cleaning:")
    print(missing_values, "\n")
    
    # Drop rows with missing income or ZIP values
    zip_cleaned: pd.DataFrame = zip_combine.dropna()
    print("Shape after dropping missing rows:", zip_cleaned.shape)
    
    # Identify and remove duplicate ZIP code entries
    duplicate_zip_count: int = zip_cleaned.duplicated(subset='ZIP').sum()
    print("Duplicate ZIP codes after cleaning:", duplicate_zip_count)
    zip_cleaned = zip_cleaned.drop_duplicates(subset='ZIP')
    print("Shape after dropping duplicates:", zip_cleaned.shape, "\n")
    
    # Standardize the 'postal_code' in the input DataFrame
    df.loc[:, 'postal_code'] = df['postal_code'].astype(str).str.zfill(5)
    
    # Merge the income data with the input DataFrame on ZIP code
    merged_df: pd.DataFrame = df.merge(zip_cleaned, left_on='postal_code', right_on='ZIP', how='inner')
    merged_df = merged_df.drop(columns=['postal_code']).rename(columns={'ZIP': 'zip_code'})
    print("Merged DataFrame shape:", merged_df.shape)
    
    return merged_df
