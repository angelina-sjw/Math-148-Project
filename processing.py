import ast
import pandas as pd

def parse_dict_col(column):
    """
    converts string to dictionary

    Returns: parsed dict or an empty dict if parsing fails.
    """
    if isinstance(column, dict):
        return column  # Already a dictionary, return as is

    try:
        parsed_dict = ast.literal_eval(column.strip('"'))  # remove extra quotes and parse
        if isinstance(parsed_dict, dict):
            return parsed_dict
        else:
            return {}
    except (ValueError, SyntaxError) as e:
        return {}  # Return empty dictionary on error


def expand_dict_col(df, column_name, expected_keys):
    """
    expands a dictionary-like column into separate boolean(0/1) columns.

    Returns: DataFrame with newly added columns.
    """
    expanded_df = pd.json_normalize(df[column_name].apply(parsing))
    expanded_df = expanded_df.reindex(columns=expected_keys, fill_value=False)

    # convert boolean values to integers
    expanded_df = expanded_df.fillna(False).astype(int)
    df = df.join(expanded_df)

    return df

def process_parking_ambience_categories(df):
  """
  expands "BusinessParking" and "Ambience" columns into separate boolean(0/1) columns.
  drop "categories" colmn

  Returns: processed DataFrame.
  """
  parking_cols = ["garage", "street", "validated", "lot", "valet"]
  df = expand_dict_col(df, "BusinessParking", parking_cols)
  ambience_cols = ["touristy", "hipster", "romantic", "divey", "intimate", "trendy", "upscale", "classy", "casual"]
  df = expand_dict_col(df, "Ambience", ambience_cols)
  df = df.drop(columns=["categories"])
  return df

# Usage Example 
# file_path = "yelp_sampled_data.csv"
# df = pd.read_csv(file_path)
# df = process_parking_ambience_categories(df)
# df.to_csv("processed_yelp_data.csv", index=False)