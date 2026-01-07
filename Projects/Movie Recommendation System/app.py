import streamlit as st
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import joblib
import ast

# Load pre-computed models
@st.cache_data
def load_models():
    models = joblib.load('models.pkl')
    return models

models = load_models()
sig = models['sig']
cos = models['cos']
collab_sim = models['collab_sim']
hybrid_sim = models['hybrid_sim']
indices = models['indices']

# Load data
@st.cache_data
def load_data():
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits_columns = credits.rename(columns={'movie_id':'id'})
    movies_merge = movies.merge(credits_columns, on='id')
    movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
    movies_cleaned = movies_cleaned.drop_duplicates(subset=['original_title'])
    movies_cleaned['genres_list'] = movies_cleaned['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])
    return movies_cleaned

movies_cleaned = load_data()

def give_recommendations(title, sim_matrix):
    if title not in indices.index:
        closest, score = process.extractOne(title, indices.index)
        if score > 80:
            title = closest
            st.write(f"Did you mean '{title}'? (score: {score})")
        else:
            st.error(f"Movie not found. Closest match: '{closest}' (score: {score})")
            return []
    idx = indices[title]
    sig_scores = list(enumerate(sim_matrix[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:21]
    movie_indices = [i[0] for i in sig_scores]
    return movies_cleaned['original_title'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("Movie Recommendation System")
st.write("Enter a movie title to get recommendations!")

movie_title = st.text_input("Movie Title", "The Matrix")

algorithm = st.selectbox("Choose Algorithm", ["Content-Based (Sigmoid)", "Content-Based (Cosine)", "Collaborative (SVD)", "Hybrid (Cosine + SVD)"])

sim_dict = {
    "Content-Based (Sigmoid)": sig,
    "Content-Based (Cosine)": cos,
    "Collaborative (SVD)": collab_sim,
    "Hybrid (Cosine + SVD)": hybrid_sim
}

if st.button("Get Recommendations"):
    if movie_title:
        recs = give_recommendations(movie_title, sim_dict[algorithm])
        if recs:
            st.subheader(f"Recommendations using {algorithm}:")
            for i, rec in enumerate(recs[:10], 1):
                st.write(f"{i}. {rec}")
    else:
        st.warning("Please enter a movie title.")