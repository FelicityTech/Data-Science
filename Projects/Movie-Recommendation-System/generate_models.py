#!/usr/bin/env python3
"""
Model Generation Script for Movie Recommendation System

This script generates the ML models required for the movie recommendation system.
Run this script to create models.pkl after cloning the repository.

Requirements:
- pandas
- numpy
- scikit-learn
- rapidfuzz
- python-levenshtein
- joblib

Usage:
    python generate_models.py
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, sigmoid_kernel
from sklearn.decomposition import TruncatedSVD
import joblib
import ast
import os


def load_data():
    """Load and preprocess movie data"""
    print("Loading movie data...")
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')

    # Merge datasets
    credits_columns = credits.rename(columns={'movie_id': 'id'})
    movies_merge = movies.merge(credits_columns, on='id')

    # Clean data
    movies_cleaned = movies_merge.drop(columns=[
        'homepage', 'title_x', 'title_y', 'status', 'production_countries'
    ])
    movies_cleaned = movies_cleaned.drop_duplicates(subset=['original_title'])

    # Process genres
    movies_cleaned['genres_list'] = movies_cleaned['genres'].apply(
        lambda x: [i['name']
                   for i in ast.literal_eval(x)] if pd.notnull(x) else []
    )

    # Process keywords
    movies_cleaned['keywords_list'] = movies_cleaned['keywords'].apply(
        lambda x: [i['name']
                   for i in ast.literal_eval(x)] if pd.notnull(x) else []
    )

    return movies_cleaned


def create_similarity_matrices(movies_cleaned):
    """Create content-based similarity matrices"""
    print("Creating content-based similarity matrices...")

    # TF-IDF Vectorization
    tfv = TfidfVectorizer(min_df=3, stop_words='english')
    tfv_matrix = tfv.fit_transform(movies_cleaned['overview'].fillna(''))

    # Cosine Similarity
    cos_sim = cosine_similarity(tfv_matrix)

    # Sigmoid Kernel
    sig_sim = sigmoid_kernel(tfv_matrix)

    return cos_sim, sig_sim


def create_collaborative_matrix(movies_cleaned):
    """Create collaborative filtering similarity matrix"""
    print("Creating collaborative filtering matrix...")

    # Create user-item matrix (simplified version)
    # In a real scenario, you'd have actual user ratings
    # Here we create a synthetic matrix based on genres and ratings

    # Use SVD for dimensionality reduction
    svd = TruncatedSVD(n_components=150, random_state=42)

    # Create a simple user-item matrix based on movie features
    # This is a simplified approach - real collaborative filtering
    # would use actual user-movie rating data
    feature_matrix = movies_cleaned[[
        'vote_average', 'vote_count', 'popularity']].fillna(0)

    # Apply SVD
    reduced_matrix = svd.fit_transform(feature_matrix)

    # Create similarity matrix from reduced dimensions
    collab_sim = cosine_similarity(reduced_matrix)

    return collab_sim


def create_hybrid_matrix(cos_sim, collab_sim):
    """Create hybrid similarity matrix"""
    print("Creating hybrid similarity matrix...")

    # Weighted combination
    content_weight = 0.6
    collab_weight = 0.4

    hybrid_sim = (cos_sim * content_weight) + (collab_sim * collab_weight)

    return hybrid_sim


def create_indices(movies_cleaned):
    """Create movie title to index mapping"""
    print("Creating movie indices...")
    indices = pd.Series(
        movies_cleaned.index, index=movies_cleaned['original_title']).drop_duplicates()
    return indices


def main():
    """Main function to generate and save models"""
    print("Starting model generation...")

    # Load data
    movies_cleaned = load_data()

    # Create similarity matrices
    cos_sim, sig_sim = create_similarity_matrices(movies_cleaned)
    collab_sim = create_collaborative_matrix(movies_cleaned)
    hybrid_sim = create_hybrid_matrix(cos_sim, collab_sim)

    # Create indices
    indices = create_indices(movies_cleaned)

    # Save models
    print("Saving models...")
    models = {
        'sig': sig_sim,
        'cos': cos_sim,
        'collab_sim': collab_sim,
        'hybrid_sim': hybrid_sim,
        'indices': indices
    }

    joblib.dump(models, 'models.pkl')
    print(
        f"Models saved to models.pkl ({os.path.getsize('models.pkl') / (1024*1024):.2f} MB)")

    print("Model generation complete!")
    print("\nYou can now run:")
    print("  streamlit run app.py    # Web interface")
    print("  python fastapi_app.py          # API server")


if __name__ == "__main__":
    main()
