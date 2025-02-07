import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sentence_transformers import SentenceTransformer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV

file_path = "/Users/sara/Documents/25W/Math148/Code/yelp_sampled_data.csv"
df = pd.read_csv(file_path)
# convert useful votes to useful classifications
df_filtered = df.dropna(subset = ['useful', 'business_id'])

# Compute 40th and 60th percentile of "useful" per business
df_filtered['p30'] = df_filtered.groupby('business_id')['useful']\
                                .transform(lambda x: x.quantile(0.30))
df_filtered['p70'] = df_filtered.groupby('business_id')['useful']\
                                .transform(lambda x: x.quantile(0.70))

# Define classification function based on cutoffs
def classify_useful(row):
    if row['useful'] < row['p30']:
        return "less useful"
    elif row['useful'] > row['p70']:
        return "more useful"
    else:
        return "average"

df_filtered['useful_category'] = df_filtered.apply(classify_useful, axis=1)

# Remove the "average" rows to focus on a 2-class problem
df_filtered = df_filtered[df_filtered['useful_category'] != 'average']
df_filtered.drop(['p30', 'p70'], axis=1, inplace=True)

y = df_filtered['useful_category']
X = df_filtered.drop(columns=['useful', 'useful_category'])
categorical_cols = ['state', 'city', 'categories', 'BusinessParking', 'Ambience', 'RestaurantsAttire']
numerical_cols = ['stars', 'RestaurantsPriceRange2', 'cool', 'business_stars', 'business_review_count',
                  'user_review_count', 'user_useful', 'user_funny', 'user_cool', 'average_stars', 'fans']

# Sentiment analysis
df_filtered['sentiment'] = df_filtered['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
numerical_cols.append('sentiment')

# SBERT embeddings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
text_embeddings_sbert = sbert_model.encode(df_filtered['text'].tolist(), convert_to_numpy = True)
pca_sbert = PCA(n_components = 50, random_state = 42)
text_embeddings_sbert_reduced = pca_sbert.fit_transform(text_embeddings_sbert)
embedding_cols_sbert = [f"text_sbert_pca_{i}" for i in range(50)]
df_embeddings_sbert = pd.DataFrame(text_embeddings_sbert_reduced, columns = embedding_cols_sbert, index = df_filtered.index)
df_filtered = pd.concat([df_filtered, df_embeddings_sbert], axis=1)
numerical_cols.extend(embedding_cols_sbert)

# LDA topic modeling
vectorizer = CountVectorizer(max_features = 1000, stop_words = 'english')
X_text = vectorizer.fit_transform(df_filtered['text'])
lda = LatentDirichletAllocation(n_components = 5, random_state = 42)
lda_topics = lda.fit_transform(X_text)
topic_cols = [f"topic_{i}" for i in range(5)]
df_topics = pd.DataFrame(lda_topics, columns = topic_cols, index = df_filtered.index)
df_filtered = pd.concat([df_filtered, df_topics], axis=1)
numerical_cols.extend(topic_cols)

# TF-IDF
tfidf = TfidfVectorizer(max_features = 500, stop_words = 'english')
X_tfidf = tfidf.fit_transform(df_filtered['text'])
tfidf_cols = [f"tfidf_{word}" for word in tfidf.get_feature_names_out()]
df_tfidf = pd.DataFrame(X_tfidf.toarray(), columns = tfidf_cols, index = df_filtered.index)
df_filtered = pd.concat([df_filtered, df_tfidf], axis = 1)
numerical_cols.extend(tfidf_cols)

X = df_filtered.drop(columns = ['text'])
# Preprocessing pipelines
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = 'most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown = 'ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

X_encoded = preprocessor.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)
# Define the hyperparameter grid
param_grid = {
    'n_estimators': [100, 200, 300, 500],  # Number of trees in the forest
    'max_depth': [10, 20, 30, None],       # Depth of each tree
    'min_samples_split': [2, 5, 10],       # Minimum samples needed to split a node
    'min_samples_leaf': [1, 2, 4],         # Minimum samples per leaf
    'max_features': ['sqrt', 'log2'],      # Number of features considered for splits
    'bootstrap': [True, False]             # Bootstrap sampling for trees
}

# Initialize Random Forest with class weights
rf_classifier = RandomForestClassifier(random_state=42, class_weight={'average': 1, 'less useful': 20, 'more useful': 20})

# Perform hyperparameter tuning using RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=rf_classifier,
    param_distributions=param_grid,
    n_iter=15,  # Number of random combinations to try
    cv=3,       # 3-fold cross-validation
    scoring='f1_weighted',  # Optimize for F1 score due to class imbalance
    random_state=42,
    n_jobs=-1   # Use all available processors
)

# Fit the model with SMOTE-resampled training data
random_search.fit(X_train, y_train)

# Get the best classifier from tuning
best_classifier = random_search.best_estimator_

# Train the best classifier
best_classifier.fit(X_train, y_train)

# Make predictions
y_pred = best_classifier.predict(X_test)

# Display the best parameters
print("Best Parameters Found:", random_search.best_params_)

# Evaluate performance
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nAccuracy:", accuracy_score(y_test, y_pred))