from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from fuzzywuzzy import process
import uvicorn
from functools import lru_cache
import os
import ast
from typing import List, Dict, Optional

app = FastAPI(title="Advanced Movie Recommendation API", description="API with explainable recommendations and advanced features")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and data
try:
    models = joblib.load('models.pkl')
    sig = models['sig']
    cos = models['cos']
    collab_sim = models['collab_sim']
    hybrid_sim = models['hybrid_sim']
    indices = models['indices']
    
    # Load movie data for details
    credits = pd.read_csv('tmdb_5000_credits.csv')
    movies = pd.read_csv('tmdb_5000_movies.csv')
    credits_columns = credits.rename(columns={'movie_id':'id'})
    movies_merge = movies.merge(credits_columns, on='id')
    movies_cleaned = movies_merge.drop(columns=['homepage', 'title_x', 'title_y', 'status', 'production_countries'])
    movies_cleaned['genres_list'] = movies_cleaned['genres'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])
    movies_cleaned['keywords_list'] = movies_cleaned['keywords'].apply(lambda x: [i['name'] for i in ast.literal_eval(x)] if pd.notnull(x) else [])
    
    print("Models and data loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"Required file not found: {e}. Please ensure models.pkl and CSV files exist.")

@lru_cache(maxsize=1000)
def get_movie_details_from_dataset(title: str) -> Dict:
    """Get movie details from our dataset"""
    if title not in indices.index:
        return {}
    
    idx = indices[title]
    movie = movies_cleaned.iloc[idx]
    
    return {
        "title": movie['original_title'],
        "overview": movie['overview'],
        "genres": movie['genres_list'],
        "keywords": movie['keywords_list'],
        "vote_average": movie['vote_average'],
        "vote_count": movie['vote_count'],
        "release_date": movie['release_date'],
        "runtime": movie['runtime'],
        "budget": movie['budget'],
        "revenue": movie['revenue']
    }

def explain_recommendation(input_title: str, rec_title: str) -> Dict:
    """Explain why a movie is recommended"""
    input_details = get_movie_details_from_dataset(input_title)
    rec_details = get_movie_details_from_dataset(rec_title)
    
    if not input_details or not rec_details:
        return {"explanation": "Details not available"}
    
    common_genres = set(input_details['genres']).intersection(set(rec_details['genres']))
    common_keywords = set(input_details['keywords']).intersection(set(rec_details['keywords']))
    
    explanation = []
    if common_genres:
        explanation.append(f"Shares genres: {', '.join(common_genres)}")
    if common_keywords:
        explanation.append(f"Common themes: {', '.join(list(common_keywords)[:3])}")  # Limit to 3
    if not explanation:
        explanation.append("Based on overall content similarity")
    
    return {
        "common_genres": list(common_genres),
        "common_keywords": list(common_keywords),
        "explanation": "; ".join(explanation)
    }

def give_recommendations(title: str, sim_matrix: np.ndarray, top_n: int = 10, include_explanations: bool = True):
    if title not in indices.index:
        closest, score = process.extractOne(title, indices.index)
        if score > 80:
            title = closest
        else:
            raise HTTPException(status_code=404, detail=f"Movie not found. Closest match: '{closest}' (score: {score})")
    
    idx = indices[title]
    sig_scores = list(enumerate(sim_matrix[idx]))
    sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
    sig_scores = sig_scores[1:top_n+1]  # Skip self
    movie_indices = [i[0] for i in sig_scores]
    recommendations = indices.iloc[movie_indices].index.tolist()
    
    if include_explanations:
        detailed_recs = []
        for rec in recommendations:
            details = get_movie_details_from_dataset(rec)
            explanation = explain_recommendation(title, rec)
            detailed_recs.append({
                "title": rec,
                **details,
                **explanation
            })
        return detailed_recs
    
    return recommendations

@app.get("/")
def read_root():
    """
    Get API information and available endpoints.

    Returns:
        dict: API metadata including available endpoints
    """
    return {"message": "Advanced Movie Recommendation API", "endpoints": ["/recommend", "/algorithms", "/movie/{title}", "/search"]}

@app.get("/recommend")
def recommend_movie(title: str, algorithm: str = "hybrid", top_n: int = 10, include_details: bool = True):
    sim_dict = {
        "sigmoid": sig,
        "cosine": cos,
        "collaborative": collab_sim,
        "hybrid": hybrid_sim
    }
    
    if algorithm not in sim_dict:
        raise HTTPException(status_code=400, detail="Invalid algorithm. Choose from: sigmoid, cosine, collaborative, hybrid")
    
    recommendations = give_recommendations(title, sim_dict[algorithm], top_n, include_details)
    
    input_details = get_movie_details_from_dataset(title)
    
    return {
        "input_movie": {
            "title": title,
            **input_details
        },
        "algorithm": algorithm,
        "recommendations": recommendations
    }

@app.get("/algorithms")
def list_algorithms():
    return {
        "algorithms": ["sigmoid", "cosine", "collaborative", "hybrid"],
        "descriptions": {
            "sigmoid": "Content-based using sigmoid kernel",
            "cosine": "Content-based using cosine similarity",
            "collaborative": "Collaborative filtering with SVD",
            "hybrid": "Combination of cosine and collaborative"
        }
    }

@app.get("/movie/{title}")
def get_movie_info(title: str):
    """Get detailed information about a specific movie"""
    details = get_movie_details_from_dataset(title)
    if not details:
        raise HTTPException(status_code=404, detail="Movie not found")
    
    return details

@app.get("/search")
def search_movies(query: str, genre: Optional[str] = None, min_rating: Optional[float] = None, limit: int = 20):
    """Search movies with optional filters"""
    # Fuzzy search for titles
    matches = process.extract(query, indices.index, limit=limit*2)
    results = []
    
    for match_title, score in matches:
        if score < 60:  # Skip low matches
            continue
        
        details = get_movie_details_from_dataset(match_title)
        if not details:
            continue
        
        # Apply filters
        if genre and genre not in details.get('genres', []):
            continue
        if min_rating and details.get('vote_average', 0) < min_rating:
            continue
        
        results.append({
            "title": match_title,
            "match_score": score,
            **details
        })
        
        if len(results) >= limit:
            break
    
    return {"query": query, "results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)