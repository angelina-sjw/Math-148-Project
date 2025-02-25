import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

def impute_binary_columns(X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, 
                          binary_cols: list) -> tuple:
    """
    Impute missing values in specified binary columns using a KNN classifier.

    For each binary column, the function:
      - Uses rows with known values to train a KNN classifier
      - Evaluates the model via stratified 5-fold cross-validation
      - Imputes missing values in both training and test datasets
    """
    # Exclude binary columns and the 'text' column from predictors.
    predictor_cols = [col for col in X_train.columns if col not in binary_cols + ['text']]

    for target in binary_cols:
        print(f"\nImputing missing values for: {target}")
        
        # Select rows where the target value is known
        known_train = X_train[X_train[target].notna()]
        X_train_known = known_train[predictor_cols]
        y_train_known = known_train[target]

        # Set up KNN and stratified 5-fold cross-validation
        knn = KNeighborsClassifier(n_neighbors=5)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(knn, X_train_known, y_train_known, cv=cv, scoring='accuracy')
        
        print(f"CV scores for {target}: {cv_scores}")
        print(f"Mean CV accuracy for {target}: {np.mean(cv_scores):.4f}")

        # Train the KNN classifier on known training data
        knn.fit(X_train_known, y_train_known)

        # Impute missing values in training data
        missing_train_idx = X_train[X_train[target].isna()].index
        if not missing_train_idx.empty:
            imputed_train = knn.predict(X_train.loc[missing_train_idx, predictor_cols])
            X_train.loc[missing_train_idx, target] = imputed_train

        # Impute missing values in test data
        missing_test_idx = X_test[X_test[target].isna()].index
        if not missing_test_idx.empty:
            imputed_test = knn.predict(X_test.loc[missing_test_idx, predictor_cols])
            X_test.loc[missing_test_idx, target] = imputed_test

        print(f"Remaining missing values in train for {target}: {X_train[target].isna().sum()}")
        print(f"Remaining missing values in test for {target}: {X_test[target].isna().sum()}")

    return X_train, X_test