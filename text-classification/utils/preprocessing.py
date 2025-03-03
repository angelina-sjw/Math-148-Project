import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

### Data Loading and Cleaning ###
def load_and_clean_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV file and clean the data by dropping unnecessary columns 
    and converting specified binary columns to numeric categorical codes.
    """
    df = pd.read_csv(file_path)
    cols_to_drop = ['cool', 'business_id', 'zip_code', 'user_id', 'useful', 'ambience_count']
    df.drop(columns=cols_to_drop, inplace=True)
    for col in ['price_binary', 'attire_binary']:
        df[col] = df[col].astype('category').cat.codes.replace(-1, np.nan)
    return df

### Numeric Transformations ###
def transform_numeric_features(X_train: pd.DataFrame,
                               X_test: pd.DataFrame,
                               skewed_features: list,
                               numeric_features: list,
                               stars_feature: list) -> tuple:
    """
    Apply log transformation and scaling to skewed features and standard scaling
    to other numeric features.
    """
    skewed_pipeline = Pipeline([
        ("log_transform", FunctionTransformer(np.log1p, validate=True)),
        ("scaler", StandardScaler())
    ])
    X_train[skewed_features] = X_train[skewed_features].astype(np.float64)
    X_test[skewed_features] = X_test[skewed_features].astype(np.float64)

    X_train.loc[:, skewed_features] = skewed_pipeline.fit_transform(X_train[skewed_features])
    X_test.loc[:, skewed_features] = skewed_pipeline.transform(X_test[skewed_features])
    
    other_numeric = numeric_features + stars_feature
    numeric_pipeline = Pipeline([("scaler", StandardScaler())])
    X_train[other_numeric] = X_train[other_numeric].astype(np.float64)
    X_test[other_numeric] = X_test[other_numeric].astype(np.float64)
    X_train.loc[:, other_numeric] = numeric_pipeline.fit_transform(X_train[other_numeric])
    X_test.loc[:, other_numeric] = numeric_pipeline.transform(X_test[other_numeric])
    
    return X_train, X_test

### Imputation ###
def impute_binary_columns(X_train: pd.DataFrame, 
                          X_test: pd.DataFrame, 
                          binary_cols: list) -> tuple:
    """
    Impute missing values in specified binary columns using a KNN classifier.
    """
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    predictor_cols = [col for col in X_train.columns if col not in binary_cols + ['text']]
    for target in binary_cols:
        print(f"\nImputing missing values for: {target}")
        known_train = X_train[X_train[target].notna()]
        X_train_known = known_train[predictor_cols]
        y_train_known = known_train[target]
        
        knn = KNeighborsClassifier(n_neighbors=5)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(knn, X_train_known, y_train_known, cv=cv, scoring='accuracy')
        print(f"CV scores for {target}: {cv_scores}")
        print(f"Mean CV accuracy for {target}: {np.mean(cv_scores):.4f}")
        
        knn.fit(X_train_known, y_train_known)
        missing_train_idx = X_train[X_train[target].isna()].index
        if not missing_train_idx.empty:
            X_train.loc[missing_train_idx, target] = knn.predict(X_train.loc[missing_train_idx, predictor_cols])
        missing_test_idx = X_test[X_test[target].isna()].index
        if not missing_test_idx.empty:
            X_test.loc[missing_test_idx, target] = knn.predict(X_test.loc[missing_test_idx, predictor_cols])
        print(f"Remaining missing values in train for {target}: {X_train[target].isna().sum()}")
        print(f"Remaining missing values in test for {target}: {X_test[target].isna().sum()}")
    return X_train, X_test

### Text Feature Engineering ###
from textblob import TextBlob
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class TextTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract text features: sentiment, SBERT embeddings (with PCA),
    optional LDA topics, and Count/Tfidf features (with n-grams support).
    """
    def __init__(self, 
                 sbert_model_name: str = 'all-MiniLM-L6-v2', 
                 pca_components: int = 50, 
                 pca_threshold: float = 0.95,
                 use_lda: bool = True,
                 lda_topics: int = 5,
                 count_max_features: int = 1000, 
                 tfidf_max_features: int = 500,
                 vectorizer_type: str = 'both',  # Options: 'count', 'tfidf', 'both'
                 ngram_range: tuple = (1, 1)):
        self.sbert_model_name = sbert_model_name
        self.pca_components = pca_components
        self.pca_threshold = pca_threshold
        self.use_lda = use_lda
        self.lda_topics = lda_topics
        self.count_max_features = count_max_features
        self.tfidf_max_features = tfidf_max_features
        self.vectorizer_type = vectorizer_type
        self.ngram_range = ngram_range

    def fit(self, X: pd.DataFrame, y=None):
        # Fit SBERT model and compute embeddings
        self.sbert_model_ = SentenceTransformer(self.sbert_model_name)
        sbert_embeddings = self.sbert_model_.encode(X['text'].tolist(), convert_to_numpy=True)
        
        # Check explained variance with a full PCA
        pca_full = PCA(n_components=None, random_state=42).fit(sbert_embeddings)
        cum_var = np.cumsum(pca_full.explained_variance_ratio_)
        if self.pca_components <= len(cum_var):
            current_cum_var = cum_var[self.pca_components - 1]
        else:
            current_cum_var = 1.0
        if current_cum_var < self.pca_threshold:
            optimal_n = np.argmax(cum_var >= self.pca_threshold) + 1
            print(f"Current PCA components ({self.pca_components}) explain only {current_cum_var:.2f} variance. Increasing to {optimal_n} to reach threshold of {self.pca_threshold}.")
            self.pca_components = optimal_n
        else:
            print(f"Current PCA components ({self.pca_components}) explain {current_cum_var:.2f} variance, which meets the threshold of {self.pca_threshold}.")
        self.pca_ = PCA(n_components=self.pca_components, random_state=42).fit(sbert_embeddings)
        
        # Set up vectorizers based on vectorizer_type and ngram_range
        if self.vectorizer_type in ['count', 'both']:
            self.count_vectorizer_ = CountVectorizer(max_features=self.count_max_features, 
                                                     stop_words='english',
                                                     ngram_range=self.ngram_range)
            self.count_vectorizer_.fit(X['text'])
        if self.vectorizer_type in ['tfidf', 'both']:
            self.tfidf_vectorizer_ = TfidfVectorizer(max_features=self.tfidf_max_features, 
                                                     stop_words='english',
                                                     ngram_range=self.ngram_range)
            self.tfidf_vectorizer_.fit(X['text'])
            
        # LDA: only if enabled
        if self.use_lda:
            # Use the count_vectorizer for LDA input (create if not already exists)
            if not hasattr(self, 'count_vectorizer_'):
                self.count_vectorizer_ = CountVectorizer(max_features=self.count_max_features, 
                                                         stop_words='english',
                                                         ngram_range=self.ngram_range)
                self.count_vectorizer_.fit(X['text'])
            bow = self.count_vectorizer_.transform(X['text'])
            self.lda_ = LatentDirichletAllocation(n_components=self.lda_topics, random_state=42).fit(bow)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        X_ = X.copy()
        # Extract sentiment feature
        X_['sentiment'] = X_['text'].apply(lambda txt: TextBlob(str(txt)).sentiment.polarity)
        
        # SBERT embeddings with PCA reduction
        sbert_embeddings = self.sbert_model_.encode(X_['text'].tolist(), convert_to_numpy=True)
        sbert_reduced = self.pca_.transform(sbert_embeddings)
        sbert_cols = [f'text_sbert_pca_{i}' for i in range(self.pca_components)]
        df_sbert = pd.DataFrame(sbert_reduced, columns=sbert_cols, index=X_.index)
        X_ = pd.concat([X_, df_sbert], axis=1)
        
        # LDA topics if enabled
        if self.use_lda:
            bow = self.count_vectorizer_.transform(X_['text'])
            topics = self.lda_.transform(bow)
            topic_cols = [f'topic_{i}' for i in range(self.lda_topics)]
            df_topics = pd.DataFrame(topics, columns=topic_cols, index=X_.index)
            X_ = pd.concat([X_, df_topics], axis=1)
        
        # Text vectorizer features based on vectorizer_type
        vectorizer_dfs = []
        if self.vectorizer_type in ['count', 'both']:
            count_features = self.count_vectorizer_.transform(X_['text'])
            count_cols = [f"count_{w}" for w in self.count_vectorizer_.get_feature_names_out()]
            df_count = pd.DataFrame(count_features.toarray(), columns=count_cols, index=X_.index)
            vectorizer_dfs.append(df_count)
        if self.vectorizer_type in ['tfidf', 'both']:
            tfidf_features = self.tfidf_vectorizer_.transform(X_['text'])
            tfidf_cols = [f"tfidf_{w}" for w in self.tfidf_vectorizer_.get_feature_names_out()]
            df_tfidf = pd.DataFrame(tfidf_features.toarray(), columns=tfidf_cols, index=X_.index)
            vectorizer_dfs.append(df_tfidf)
        if vectorizer_dfs:
            df_vectorizers = pd.concat(vectorizer_dfs, axis=1)
            X_ = pd.concat([X_, df_vectorizers], axis=1)
        
        X_.drop(columns=['text'], inplace=True)
        return X_

def transform_text_features(X_train: pd.DataFrame, 
                            X_test: pd.DataFrame, 
                            y_train, 
                            text_transformer: TextTransformer, 
                            sample_size: float = 0.1) -> tuple:
    """
    Fit the text transformer on a stratified sample of X_train and transform both train and test sets.
    """
    from sklearn.model_selection import train_test_split
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train, test_size=1 - sample_size, stratify=y_train, random_state=42
    )
    text_transformer.fit(X_train_sample, y_train_sample)
    X_train_transformed = text_transformer.transform(X_train)
    X_test_transformed = text_transformer.transform(X_test)
    return X_train_transformed, X_test_transformed
