import ast
import pandas as pd
from typing import Any, List


def parse_dict_col(column: Any) -> dict:
    """
    Convert a string representation of a dictionary into an actual dict.
    
    If the input is already a dict, it is returned unchanged. Otherwise, the function
    attempts to parse the string using ast.literal_eval after stripping surrounding quotes.
    In case of failure, an empty dict is returned.
    """
    if isinstance(column, dict):
        return column

    try:
        # Remove extra quotes and attempt to evaluate the string as a dict
        parsed_dict = ast.literal_eval(column.strip('"'))
        return parsed_dict if isinstance(parsed_dict, dict) else {}
    except (ValueError, SyntaxError):
        return {}


def expand_dict_col(df: pd.DataFrame, column_name: str, expected_keys: List[str]) -> pd.DataFrame:
    """
    Expand a dictionary-like column into separate boolean (0/1) columns.
    
    Each key specified in expected_keys will be created as a new column in the DataFrame.
    Missing keys and NaN values are filled with False, and all values are converted to int.
    """
    # Apply parsing to each cell in the column
    parsed_series = df[column_name].apply(parse_dict_col)
    # Normalize the parsed dictionaries into a DataFrame
    expanded_df = pd.json_normalize(parsed_series)
    # Ensure all expected keys are present; fill missing keys with False
    expanded_df = expanded_df.reindex(columns=expected_keys, fill_value=False)
    expanded_df = expanded_df.fillna(False).astype(int)
    # Join the new columns with the original DataFrame
    return df.join(expanded_df)


def process_parking_ambience_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process and expand the "BusinessParking" and "Ambience" columns into separate boolean columns.
    
    The function creates new columns for parking (e.g., 'garage', 'street') and ambience
    (e.g., 'touristy', 'hipster') features, then drops the original "categories" column.
    """
    # Define expected keys for parking and ambience attributes
    parking_cols = ["garage", "street", "validated", "lot", "valet"]
    ambience_cols = ["touristy", "hipster", "romantic", "divey", "intimate", "trendy", "upscale", "classy", "casual"]

    # Expand the dictionary columns into separate boolean columns
    df = expand_dict_col(df, "BusinessParking", parking_cols)
    df = expand_dict_col(df, "Ambience", ambience_cols)
    
    # Drop the original 'categories' column as it's no longer needed
    return df.drop(columns=["categories"])