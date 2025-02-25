from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt  # retained for potential visualization use
import seaborn as sns           # retained for potential visualization use
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

def create_preprocessor(X_transformed):
    """
    Create a preprocessor that applies data transformations to numeric and categorical features.
    
    Numeric features are imputed (mean) and scaled, while categorical features are imputed
    (most frequent) and one-hot encoded.
    """
    # Identify numeric and categorical columns based on data types
    numeric_cols = [
        col for col in X_transformed.columns 
        if X_transformed[col].dtype in [np.float64, np.int64]
    ]
    categorical_cols = [
        col for col in X_transformed.columns 
        if X_transformed[col].dtype == 'object'
    ]
    
    # Pipeline for numeric features: impute missing values and scale
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    
    # Pipeline for categorical features: impute missing values and encode
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine the pipelines into a single ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    
    return preprocessor


def run_classification(X_train, y_train, X_test, y_test):
    """
    Perform classification using a Random Forest classifier with SMOTE for class balancing.
    
    The function applies SMOTE to balance the training data, trains a RandomForestClassifier
    with specified class weights, and prints the classification report and accuracy score.
    """
    # Apply SMOTE to balance the training dataset
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Initialize the classifier with custom class weights to handle class imbalance
    classifier = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight={'average': 1, 'less useful': 20, 'more useful': 20}
    )
    
    # Train the classifier on the resampled data
    classifier.fit(X_train_resampled, y_train_resampled)
    
    # Predict on the test data
    y_pred = classifier.predict(X_test)
    
    # Output evaluation metrics
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))
    
    return y_pred