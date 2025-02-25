import pandas as pd
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Any, Tuple

class TextTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer that extracts various text features:
      1. Computes sentiment via TextBlob.
      2. Computes SBERT embeddings and applies PCA for dimensionality reduction.
      3. Derives LDA topics from bag-of-words.
      4. Generates TF-IDF features.
      5. Drops the original text column after processing.
    """
    def __init__(self, 
                 sbert_model_name: str = 'all-MiniLM-L6-v2', 
                 pca_components: int = 50, 
                 lda_topics: int = 5,
                 count_max_features: int = 1000, 
                 tfidf_max_features: int = 500):
        self.sbert_model_name = sbert_model_name
        self.pca_components = pca_components
        self.lda_topics = lda_topics
        self.count_max_features = count_max_features
        self.tfidf_max_features = tfidf_max_features

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "TextTransformer":
        """
        Fit the text transformer by computing required models/transformers on the training data.
        """
        # Initialize the SBERT model and compute embeddings for PCA fitting.
        self.sbert_model_ = SentenceTransformer(self.sbert_model_name)
        sbert_embeddings = self.sbert_model_.encode(X['text'].tolist(), convert_to_numpy=True)
        
        # Fit PCA on SBERT embeddings for dimensionality reduction.
        self.pca_ = PCA(n_components=self.pca_components, random_state=42)
        self.pca_.fit(sbert_embeddings)
        
        # Fit CountVectorizer to create bag-of-words and then LDA for topic extraction.
        self.count_vectorizer_ = CountVectorizer(max_features=self.count_max_features, stop_words='english')
        bow = self.count_vectorizer_.fit_transform(X['text'])
        self.lda_ = LatentDirichletAllocation(n_components=self.lda_topics, random_state=42)
        self.lda_.fit(bow)
        
        # Fit TF-IDF vectorizer on the text data.
        self.tfidf_vectorizer_ = TfidfVectorizer(max_features=self.tfidf_max_features, stop_words='english')
        self.tfidf_vectorizer_.fit(X['text'])
        
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Transform the input DataFrame by extracting text features.
        
        Features extracted:
          - Sentiment polarity using TextBlob.
          - PCA-reduced SBERT embeddings.
          - LDA topic probabilities.
          - TF-IDF features.
        
        The original 'text' column is dropped after feature extraction.
        """
        X_ = X.copy()
        
        # Compute sentiment polarity for each text entry.
        X_['sentiment'] = X_['text'].apply(lambda txt: TextBlob(str(txt)).sentiment.polarity)
        
        # Generate SBERT embeddings and reduce dimensionality via PCA.
        sbert_embeddings = self.sbert_model_.encode(X_['text'].tolist(), convert_to_numpy=True)
        sbert_reduced = self.pca_.transform(sbert_embeddings)
        sbert_cols = [f'text_sbert_pca_{i}' for i in range(self.pca_components)]
        df_sbert = pd.DataFrame(sbert_reduced, columns=sbert_cols, index=X_.index)
        X_ = pd.concat([X_, df_sbert], axis=1)
        
        # Compute bag-of-words representation and extract LDA topics.
        bow = self.count_vectorizer_.transform(X_['text'])
        topics = self.lda_.transform(bow)
        topic_cols = [f'topic_{i}' for i in range(self.lda_topics)]
        df_topics = pd.DataFrame(topics, columns=topic_cols, index=X_.index)
        X_ = pd.concat([X_, df_topics], axis=1)
        
        # Generate TF-IDF features.
        tfidf = self.tfidf_vectorizer_.transform(X_['text'])
        tfidf_cols = [f"tfidf_{w}" for w in self.tfidf_vectorizer_.get_feature_names_out()]
        df_tfidf = pd.DataFrame(tfidf.toarray(), columns=tfidf_cols, index=X_.index)
        X_ = pd.concat([X_, df_tfidf], axis=1)
        
        # Drop the original text column.
        X_.drop(columns=['text'], inplace=True)
        return X_


def transform_text_features(X_train: pd.DataFrame, 
                            X_test: pd.DataFrame, 
                            y_train: pd.Series, 
                            text_transformer: TextTransformer, 
                            sample_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit a text transformer on a stratified sample of the training data and transform both
    the training and test sets.
    
    A stratified sample (by y_train) of size 'sample_size' is used for fitting the transformer.
    """
    from sklearn.model_selection import train_test_split

    # Create a stratified sample from the training data for efficient fitting.
    X_train_sample, _, y_train_sample, _ = train_test_split(
        X_train, y_train, test_size=1 - sample_size, stratify=y_train, random_state=42
    )
    
    # Fit the text transformer on the sample.
    text_transformer.fit(X_train_sample, y_train_sample)
    
    # Transform both the training and test sets.
    X_train_transformed = text_transformer.transform(X_train)
    X_test_transformed = text_transformer.transform(X_test)
    
    return X_train_transformed, X_test_transformed