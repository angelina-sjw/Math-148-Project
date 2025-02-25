# Yelp Data Processing and Analysis Pipeline

This repository contains a complete end-to-end pipeline for processing, cleaning, transforming, and analyzing Yelp review data. The project is organized into multiple modules that load and clean data, reduce dimensions, impute missing values, extract text features, build machine learning models, and visualize results. It is designed to facilitate exploration of review usefulness and supports a classification task that categorizes reviews into "less useful" or "more useful" (with "average" reviews filtered out).

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation and Dependencies](#installation-and-dependencies)
- [Data Preprocessing Pipeline](#data-preprocessing-pipeline)
- [Usage Instructions](#usage-instructions)
- [Code Overview](#code-overview)

---

## Overview

This project processes raw Yelp JSON datasets by:
- Loading the JSON files into a SQLite database.
- Merging and filtering review, user, and business data.
- Performing dimensionality reduction by aggregating columns (e.g., summing votes, compliments, and transforming categorical attributes).
- Imputing missing values in binary columns using a K-Nearest Neighbors (KNN) approach.
- Extracting numeric and text features, including sentiment, SBERT embeddings (with PCA reduction), LDA topics, and TF-IDF features.
- Creating preprocessing pipelines for numeric and categorical features.
- Training a Random Forest classifier with SMOTE for class balancing and evaluating its performance using classification metrics and confusion matrices.

---

## Project Structure

- **data_utils.py**:  
  Contains functions to load CSV data, drop unnecessary columns, convert binary columns to numeric categorical codes, and transform numeric features using pipelines (e.g., log-transformation and scaling).

- **dim_reduction.py**:  
  Processes large CSV files in chunks, performs column merging and transformation (e.g., summing vote and compliment columns, mapping price and attire attributes), converts key numeric columns, and outputs a dimension-reduced CSV file.

- **imputation_utils.py**:  
  Implements a KNN-based imputation strategy to fill in missing values in specified binary columns. It uses stratified cross-validation to assess imputation quality.

- **load_data.py**:  
  Loads Yelp JSON files for reviews, users, and businesses into a SQLite database. It creates necessary tables, extracts and transforms relevant columns, processes parking and ambience attributes, merges income data by ZIP code, and outputs a filtered CSV dataset.

- **modeling_utils.py**:  
  Provides functions to build a preprocessing pipeline (combining imputation, scaling, and one-hot encoding) and run classification using a Random Forest classifier with SMOTE for handling class imbalance.

- **plotting_utils.py**:  
  Contains functions to visualize numeric distributions (histograms), categorical distributions (bar charts), target variable distribution, and confusion matrices.

- **processing.py**:  
  Offers helper functions to parse dictionary-like columns (e.g., for parking and ambience attributes) and expand these into separate boolean features.

- **text_transformer_utils.py**:  
  Defines a custom transformer (`TextTransformer`) that:
  - Computes sentiment scores using TextBlob.
  - Generates SBERT embeddings and reduces their dimensions via PCA.
  - Extracts topics using Latent Dirichlet Allocation (LDA) from bag-of-words representations.
  - Creates TF-IDF features.
  
  The original text column is dropped after feature extraction.

- **zipcode.py**:  
  Downloads and merges US household income data (from Kaggle) by ZIP code with the Yelp dataset to incorporate regional income features.

- **main.ipynb**:  
  A Jupyter Notebook that ties everything together for initial exploration:
  - Loads the preprocessed CSV.
  - Splits the data into training and testing sets.
  - Performs imputation on binary columns.
  - Plots numeric and categorical feature distributions.
  - Transforms numeric and text features.
  - Prepares the final preprocessed data using a `ColumnTransformer`.
  - Trains a Random Forest classifier (with SMOTE for balancing) and evaluates model performance, including plotting the confusion matrix.

---

## Installation and Dependencies

Ensure you have Python 3.7 or later installed. The following Python packages are required:

- pandas
- numpy
- scikit-learn
- imblearn
- matplotlib
- seaborn
- sentence_transformers
- textblob
- sqlite3 (standard library)
- kagglehub (for downloading the income dataset)
- ast (standard library)

You can install most dependencies using pip:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn sentence_transformers textblob kagglehub
```

Additional setup (e.g., for Kaggle API credentials) may be necessary to download datasets.

---

## Data Preprocessing Pipeline

1. **Loading and Cleaning Data**:  
   Run `load_data.py` to:
   - Create and populate SQLite tables from Yelp JSON files.
   - Merge review, user, and business data.
   - Process parking/ambience categories and merge income data.
   - Filter reviews based on review counts and usefulness.
   - Save the filtered merged dataset as `yelp_merged_data_filtered.csv`.

2. **Dimension Reduction**:  
   Run `dim_reduction.py` to:
   - Process the large filtered CSV in chunks.
   - Merge/transform columns (e.g., summing votes, compliments, parking, ambience).
   - Map price and attire categories.
   - Output a reduced-dimension CSV file (`yelp_reduced.csv`).

3. **Exploratory Analysis and Modeling**:  
   Open and run the `main.ipynb` notebook:
   - Load the reduced CSV.
   - Split the data into training and testing sets.
   - Perform imputation on binary columns.
   - Visualize distributions of numeric and categorical features.
   - Transform numeric features (e.g., log transformations and scaling).
   - Extract text features (sentiment, SBERT-PCA, LDA topics, TF-IDF).
   - Preprocess final features using a `ColumnTransformer`.
   - Train a Random Forest classifier (with SMOTE for balancing) and evaluate model performance.

---

## Usage Instructions

1. **Prepare the Data:**
   - First, run `load_data.py` to load raw Yelp JSON files into SQLite and output the merged CSV.
   - Then, run `dim_reduction.py` to reduce data dimensions and produce the `yelp_reduced.csv` file.

2. **Explore and Model:**
   - Open `main.ipynb` in Jupyter Notebook.
   - Follow the notebook cells to load data, perform train-test splits, impute missing values, visualize data, transform features, build preprocessing pipelines, and run the classification model.

3. **Visualization and Evaluation:**
   - Use the provided plotting functions to explore feature distributions.
   - Review the classification report and confusion matrix generated from the model evaluation.

---

## Code Overview

Each module in the project plays a key role in the pipeline:

- **Data Loading & Cleaning**:  
  `data_utils.py` and `load_data.py` manage reading and cleaning data from CSV and JSON formats.
  
- **Dimension Reduction & Feature Engineering**:  
  `dim_reduction.py`, `processing.py`, and `zipcode.py` focus on reducing the data size and creating new features through column aggregation and external income data merging.

- **Imputation & Transformation**:  
  `imputation_utils.py` handles missing binary values via KNN, while `text_transformer_utils.py` extracts rich text features.

- **Modeling & Visualization**:  
  `modeling_utils.py` builds the ML model pipeline and applies SMOTE, and `plotting_utils.py` provides various plotting utilities for exploratory data analysis.

---