# imports
import pickle
# from sklearn.base import BaseEstimator, ClusterMixin
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from tqdm import tqdm
import cudf
from cuml.feature_extraction.text import TfidfVectorizer as cuTfidfVectorizer
from cuml.cluster import DBSCAN as cumlDBSCAN, KMeans as cumlKMeans
from collections import Counter
import pandas as pd
from joblib import Parallel, delayed
from sklearn.preprocessing import MultiLabelBinarizer
# import unicodedata
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
# import fasttext
import json
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from collections import Counter
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances
# from gensim.models import KeyedVectors
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import os
import joblib

class DBSCANPipeline():
    def __init__(self, eps, min_samples, feature_matrix, rapid=True, random_state=0):
        self.eps = eps
        self.min_samples = min_samples
        self.random_state = random_state
        self.feature_matrix = feature_matrix
        if rapid:
            self.model = cumlDBSCAN(eps=eps, min_samples=min_samples)
        else:
            self.model = DBSCAN(eps=eps, min_samples=min_samples)
        
    def fit(self):
        self.labels_ = self.model.fit_predict(self.feature_matrix)
        return self.labels_
    
    
    def evaluate_clusters(self):
        if not hasattr(self, 'labels_'):
            raise ValueError("Model has not been fit yet. Please call the fit method first.")
            # Unique clusters (excluding noise points labeled as -1)
        unique_clusters = set(self.labels_)
        unique_clusters.discard(-1)  # Remove noise label
        n_clusters = len(unique_clusters)
        
        print(f"Number of clusters (excluding noise): {n_clusters}")
        print(f"Number of noise points: {(self.labels_ == -1).sum()}")
        
        # If there's more than one cluster, calculate silhouette score
        if n_clusters > 1:
            silhouette_avg = silhouette_score(self.feature_matrix, self.labels_)
            print(f"Silhouette Score: {silhouette_avg:.4f}")
        
        # Calculate intra-cluster distances (using medoids)
        cluster_metrics = []
        for cluster in unique_clusters:
            cluster_points = self.feature_matrix[self.labels_ == cluster]
            medoid = cluster_points.mean(axis=0)  # Use the mean as a medoid approximation
            intra_distances = np.linalg.norm(cluster_points - medoid, axis=1)
            cluster_metrics.append({
                'Cluster': cluster,
                'Intra-cluster Median Distance': np.median(intra_distances),
                'Intra-cluster Mean Distance': np.mean(intra_distances)
            })
        
        # Create a DataFrame for cluster metrics
        cluster_metrics_df = pd.DataFrame(cluster_metrics)
        print("Cluster Metrics:\n", cluster_metrics_df)
    
    def compute_cluster_averages(self, classification_feature_matrix):
        """Compute the average feature vectors for each cluster, using the feature_matrix and cluster labels."""
        if not hasattr(self, 'labels_'):
            raise ValueError("Model has not been fit yet. Please call the fit method first.")
        # Initialize a dictionary to store the average feature vector for each cluster
        cluster_averages = {}
        # Loop over each unique cluster label
        for cluster_label in np.unique(self.labels_):
            # Get the indices of businesses in the current cluster
            cluster_indices = np.where(self.labels_ == cluster_label)[0]
            # Extract the feature vectors for these businesses from the classification feature matrix
            cluster_data = classification_feature_matrix[cluster_indices]
            # Compute the average of the feature vectors for this cluster
            cluster_averages[cluster_label] = np.mean(cluster_data, axis=0)
        # Store the cluster averages in an attribute for future reference
        self.cluster_averages = cluster_averages

        return cluster_averages
    
    def cluster_insights(self, business_data, classification_features, classification_feature_matrix):
        """
        Generate insights for each cluster, including sentiment analysis and keyword extraction.
        Utilizes RAPIDS cuDF and cuML for GPU acceleration.
        
        Args:
            business_data (pd.DataFrame): DataFrame containing business data.
            classification_feature_matrix (np.ndarray): Feature matrix with classification features.
        """
        # Convert business_data to cuDF DataFrame
        # Select only the relevant columns
        columns_to_keep = ['stars', 'business_id', 'latitude', 'longitude', 'review_count', 'cluster']
        business_data = business_data[columns_to_keep].copy()
        
        # Ensure consistent data types for these columns
        business_data['business_id'] = business_data['business_id'].astype(str)
        business_data['stars'] = business_data['stars'].astype(float)
        business_data['latitude'] = business_data['latitude'].astype(float)
        business_data['longitude'] = business_data['longitude'].astype(float)
        business_data['review_count'] = business_data['review_count'].astype(int)
        business_data['cluster'] = business_data['cluster'].astype(int)
        
        for column in business_data.columns:
            print(f"{column}: {business_data[column].apply(type).unique()}")
    
        business_data = cudf.from_pandas(business_data)
        
        if 'sentiment' in classification_features:
            # Add sentiment scores from the classification feature matrix
            sentiment_scores = cudf.DataFrame(classification_feature_matrix[:, -2:], columns=['prob_negative', 'prob_positive'])
            business_data = cudf.concat([business_data.reset_index(drop=True), sentiment_scores.reset_index(drop=True)], axis=1)

            # Aggregate cluster insights
            profile = business_data.groupby('cluster').agg({
                'stars': 'mean',
                'business_id': 'count',
                'latitude': 'mean',
                'longitude': 'mean',
                'review_count': 'sum',
                'prob_negative': 'mean',
                'prob_positive': 'mean'
            }).reset_index().rename(columns={
                'stars': 'average_rating',
                'business_id': 'business_count',
                'latitude': 'average_latitude',
                'longitude': 'average_longitude',
                'review_count': 'review_count',
                'prob_negative': 'avg_prob_negative',
                'prob_positive': 'avg_prob_positive'
            })
        else:
            # Aggregate cluster insights without sentiment
            profile = business_data.groupby('cluster').agg({
                'stars': 'mean',
                'latitude': 'mean',
                'longitude': 'mean',
                'business_id': 'count',
                'review_count': 'sum'
            }).reset_index().rename(columns={
                'stars': 'average_rating',
                'latitude': 'average_latitude',
                'longitude': 'average_longitude',
                'business_id': 'business_count',
                'review_count': 'review_count'
            })

        # Extract top keywords using RAPIDS cuML TF-IDF
        # profile['keywords'] = self.extract_top_keywords_with_rapids(business_data, text_column='text', top_n=10)

        print("Cluster Profile with Sentiment:\n", profile.to_pandas())
        self.cluster_profile = profile.to_pandas()  # Save as pandas for compatibility
        return self.cluster_profile

    def classify_by_closest_cluster(self, business, cluster_averages, classification_features, vectorizer, scaler=None):
        """
        Classify a given business by determining the closest cluster based on average feature vectors.

        Args:
            business (pd.Series): The data of the business to classify.

        Returns:
            int: The cluster label of the closest cluster.
        """
        if not hasattr(self, 'cluster_averages'):
            raise ValueError("Cluster averages have not been computed. Fit the model first.")

        
        # Construct the classification feature vector for the given business
        classification_vector = np.array([vectorizer.vectorize_business(business, classification_features)])
        if scaler:
            classification_vector = scaler.transform(classification_vector)

        # Find the closest cluster by computing distances to cluster averages
        closest_cluster = None
        min_distance = float('inf')
        for cluster_label, cluster_average in cluster_averages.items():
            distance = np.linalg.norm(classification_vector - cluster_average)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_label

        return closest_cluster
    
    def save(self, business_data, classification_features, classification_feature_matrix, path):
        result = dict()
        result['eps'] = self.eps
        result['min_samples'] = self.min_samples
        result['avg_features'] = self.compute_cluster_averages(classification_feature_matrix)
        result['insights'] = self.cluster_insights(business_data, classification_features, classification_feature_matrix)
        result['classification_features'] = classification_features
        result['classify_by_closest_cluster'] = self.classify_by_closest_cluster

        with open(path, 'wb') as f:
            pickle.dump(result, f)


class Vectorizer:
    def __init__(self):
        # Instance-level attributes (instead of class-level)
        self.sentiment_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.sentiment_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        self.top_categories = None
        self.kinds = ['location', 'hours', 'categories', 'stars', 'sentiment']

    def category_vector(self, business):
        business_categories = set(business['categories'].split(', ')) if isinstance(business['categories'], str) else set()
        vector = np.zeros(len(self.top_categories))
        for category in business_categories.intersection(self.top_categories):
            idx = self.top_categories.index(category)
            vector[idx] = 1
        return vector
    
    def hours_vector(self, business):
        if not business['hours']:
            return np.zeros(2)
        open_weekends = int('Saturday' in business['hours'] or 'Sunday' in business['hours'])
        open_late = int(any(h and h.split('-')[1] >= '21:00' for h in business['hours'].values()))
        return np.array([open_weekends, open_late])
    
    def location_vector(self, business):
        return np.array([business['latitude'], business['longitude']])
    
    def stars_vector(self, business):
        return np.array([business['average_star_rating']])
    
    def sentiment_vector(self, text, max_length=512, batch_size=64):
        """
        Compute sentiment vector for a large text by processing it in chunks.

        Args:
            text (str): The input text blob.
            max_length (int): Maximum token length per chunk.
            batch_size (int): Number of chunks to process in a batch.

        Returns:
            np.ndarray: Aggregated sentiment probabilities [prob_negative, prob_positive].
        """
        if not text or not isinstance(text, str):
            return np.zeros(2)  # Return neutral sentiment for empty/missing text

        # Split text into chunks based on tokenizer's max length
        tokens = self.sentiment_tokenizer(text, return_tensors="pt", truncation=False, padding=False).input_ids[0]
        chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        
        all_probabilities = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_model.to(device)
        
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]

            # Prepare batch for model
            batch_inputs = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(chunk) for chunk in batch_chunks], 
                batch_first=True
            ).to(device)

            with torch.no_grad():
                # Process batch and compute logits
                outputs = self.sentiment_model(input_ids=batch_inputs)
                logits = outputs.logits

            # Convert logits to probabilities
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            all_probabilities.extend(probabilities)

        # Aggregate probabilities across all chunks (e.g., averaging)
        aggregated_probabilities = np.mean(all_probabilities, axis=0)
        return aggregated_probabilities
    
    def save_top_categories(self, business_data):
        mlb = MultiLabelBinarizer()
        categories_df = business_data['categories'].fillna('').apply(lambda x: x.split(', '))
        categories_matrix = mlb.fit_transform(categories_df)
        top_categories = [cat for cat, _ in Counter(mlb.classes_).most_common(10)]
        self.top_categories = top_categories
        
        # Save the top_categories to a file
        if not os.path.exists('./clustering/top_categories.pkl'):
            with open('./clustering/top_categories.pkl', 'wb') as f:
                pickle.dump(self.top_categories, f)

    def vectorize_business(self, business, selected_features):
        if not self.top_categories and os.path.exists('./clustering/top_categories.pkl'):
            with open('./clustering/top_categories.pkl', 'rb') as f:
                self.top_categories = pickle.load(f)
        if not self.top_categories:
            raise ValueError("Top categories not set. Call `build_feature_matrix` first.")
        vector = []
        if 'location' in selected_features:
            vector.append(self.location_vector(business))
        if 'hours' in selected_features:
            vector.append(self.hours_vector(business))
        if 'categories' in selected_features:
            vector.append(self.category_vector(business))
        if 'stars' in selected_features:
            vector.append(self.stars_vector(business))
        if 'sentiment' in selected_features:
            vector.append(self.sentiment_vector(business.get('text', '')))
        return np.hstack(vector)
    
    def build_feature_matrix(self, business_data, selected_features, parallelize=True):
        if not self.top_categories:
            self.save_top_categories(business_data)
        if parallelize:
            feature_matrix = Parallel(n_jobs=-1)(
                delayed(self.vectorize_business)(business, selected_features)
                for _, business in tqdm(business_data.iterrows(), total=business_data.shape[0], desc="Processing Businesses")
            )
        else:
            feature_matrix = []
            for _, business in tqdm(business_data.iterrows(), total=len(business_data), desc="Building feature matrix"):
                vector = self.vectorize_business(business, selected_features)
                feature_matrix.append(vector)

        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        return feature_matrix, feature_matrix_scaled, scaler
    
    def combine_cached_feature_matrices(self, matrix_cache, selected_features):
        feature_matrices = [matrix_cache[k] for k in selected_features]
        combined = np.hstack(feature_matrices)
        scaler = StandardScaler()
        combined_scaled = scaler.fit_transform(combined)
        return combined, combined_scaled, scaler
    

# def load(filename):
#     """Load a clustering pipeline"""
#     with open(filename, 'rb') as f:
#         return pickle.load(f)
# stored_classifier = load('./classifiers/stars(0.017-500)-sentiment.pkl')
# stored_vectorizer = load('six_eye.pkl')
# top_categories = load('top_categories.pkl')

# # --- Example Usage: Define a New Business ---
# new_business = {
#     'text': "good",
#     'categories': "Cafes, Coffee, Breakfast",
#     'latitude': 37.02,
#     'longitude': -88.8104,
#     'city': "San Francisco",
#     'hours': {
#         "Monday": "07:00-18:00",
#         "Tuesday": "07:00-18:00",
#         "Wednesday": "07:00-18:00",
#         "Thursday": "07:00-18:00",
#         "Friday": "07:00-18:00",
#         "Saturday": "08:00-16:00",
#         "Sunday": "08:00-16:00"
#     },
#     "stars": None,
# }


# classification = stored_classifier['classify_by_closest_cluster'](new_business, stored_classifier['avg_features'], stored_classifier['classification_features'], stored_vectorizer)
# stored_classifier['insights'][stored_classifier['insights']['cluster'] == classification]