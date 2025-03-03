import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def create_preprocessor(X):
    """
    Create a preprocessor that imputes and scales numeric features and 
    one-hot encodes categorical features.
    """
    numeric_cols = [col for col in X.columns if X[col].dtype in [np.float64, np.int64]]
    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numeric_cols),
        ('cat', categorical_pipeline, categorical_cols)
    ])
    return preprocessor

def run_classification(X_train, y_train, X_test, y_test, regularize=False):
    """
    Run a RandomForestClassifier with optional regularization and SMOTE-based oversampling.
    """
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    
    if regularize:
        clf = RandomForestClassifier(n_estimators=20, max_depth=10, min_samples_split=20,
                                     min_samples_leaf=10, random_state=42)
    else:
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train_res, y_train_res)
    
    y_pred_train = clf.predict(X_train_res)
    print("\nTrain Classification Report:\n", classification_report(y_train_res, y_pred_train, digits=4))
    y_pred_test = clf.predict(X_test)
    print("\nTest Classification Report:\n", classification_report(y_test, y_pred_test, digits=4))
    
    return y_pred_test

def grid_search_classification(X_train, y_train):
    """
    Perform grid search with a pipeline including an optional sampler, PCA, and RandomForestClassifier.
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.decomposition import PCA
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import make_scorer, f1_score
    
    param_grid = [
        {
            'sampler': [RandomUnderSampler(random_state=42)],
            'pca': [PCA()],
            'pca__n_components': [10, 50, 100],
            'clf__n_estimators': [50, 80],
            'clf__max_depth': [2, 5, 10],
            'clf__min_samples_split': [10],
            'clf__class_weight': [None, 'balanced']
        },
        {
            'sampler': ['passthrough'],
            'pca': [PCA()],
            'pca__n_components': [10, 50, 100],
            'clf__n_estimators': [50, 80],
            'clf__max_depth': [2, 5, 10],
            'clf__min_samples_split': [10],
            'clf__class_weight': [None, 'balanced']
        }
    ]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scorer = make_scorer(f1_score, average='weighted')
    pipeline = ImbPipeline([
        ('sampler', 'passthrough'),
        ('pca', 'passthrough'),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scorer, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print("Best parameters found:", grid_search.best_params_)
    print("Best CV weighted F1 score:", grid_search.best_score_)
    
    return grid_search.best_estimator_
