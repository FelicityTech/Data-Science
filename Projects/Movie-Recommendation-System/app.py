import streamlit as st
import pandas as pd
import numpy as np
from rapidfuzz import process
import joblib
import ast
from pathlib import Path

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

# Load pre-computed models


@st.cache_resource
def load_models():
    return joblib.load(BASE_DIR / "models.pkl")


models = load_models()
sig = models['sig']
cos = models['cos']
collab_sim = models['collab_sim']
hybrid_sim = models['hybrid_sim']
indices = models['indices']

# Load data


@st.cache_data
def load_data():
    credits = pd.read_csv(BASE_DIR / 'tmdb_5000_credits.csv')
    movies = pd.read_csv(BASE_DIR / 'tmdb_5000_movies.csv')
    credits_columns = credits.rename(columns={'movie_id': 'id'})
    movies_merge = movies.merge(credits_columns, on='id')
    movies_cleaned = movies_merge.drop(
        columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
    movies_cleaned = movies_cleaned.drop_duplicates(subset=['original_title'])
    movies_cleaned['genres_list'] = movies_cleaned['genres'].apply(
        lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])
    return movies_cleaned


movies_cleaned = load_data()


def give_recommendations(title, sim_matrix, top_n=10):
    if title not in indices.index:
        result = process.extractOne(title, indices.index)
        closest, score = result[0], result[1]
        if score > 80:
            title = closest
            st.write(f"Did you mean '{title}'? (score: {score})")
        else:
            st.error(
                f"Movie not found. Closest match: '{closest}' (score: {score})")
            return []
    idx = indices[title]
    sig_scores = list(enumerate(sim_matrix[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:top_n+1]
    movie_indices = [i[0] for i in sig_scores]
    return movies_cleaned['original_title'].iloc[movie_indices].tolist()


# App Title
st.title("🎬 Movie Recommendation System")
st.markdown(
    "Discover your next favorite movie with our advanced recommendation algorithms!")

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    algorithm = st.selectbox(
        "Choose Recommendation Algorithm",
        ["Hybrid", "Content-Based (Sigmoid)",
         "Content-Based (Cosine)", "Collaborative"],
        index=0
    )
    top_n = st.slider("Number of Recommendations",
                      min_value=5, max_value=20, value=10)

# Main content
st.header("🔍 Find Recommendations")
movie_title = st.text_input("Enter a movie title:",
                            placeholder="e.g., The Matrix")

if st.button("Get Recommendations", type="primary"):
    if movie_title.strip():
        with st.spinner("Finding recommendations..."):
            # Map algorithm to sim_matrix
            if algorithm == "Hybrid":
                sim_matrix = hybrid_sim
            elif algorithm == "Content-Based (Sigmoid)":
                sim_matrix = sig
            elif algorithm == "Content-Based (Cosine)":
                sim_matrix = cos
            else:  # Collaborative
                sim_matrix = collab_sim

            recommendations = give_recommendations(
                movie_title, sim_matrix, top_n)

            if recommendations:
                st.success(
                    f"Found {len(recommendations)} recommendations for '{movie_title}'!")

                # Display recommendations in columns
                cols = st.columns(2)
                for i, rec in enumerate(recommendations):
                    with cols[i % 2]:
                        st.subheader(f"🎥 {rec}")
                        # Add more details if available
                        movie_info = movies_cleaned[movies_cleaned['original_title'] == rec]
                        if not movie_info.empty:
                            genres = movie_info['genres_list'].iloc[0]
                            if genres:
                                st.write(f"**Genres:** {', '.join(genres)}")
                            vote_avg = movie_info['vote_average'].iloc[0]
                            st.write(f"**Rating:** {vote_avg}/10")
            else:
                st.error("No recommendations found. Please try a different title.")
    else:
        st.warning("Please enter a movie title.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and advanced ML algorithms.")
