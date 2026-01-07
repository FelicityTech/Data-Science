import pytest
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import ast

def create_mock_movie_data(n_movies=10):
    """Create mock movie data for testing"""
    np.random.seed(42)  # For reproducible tests

    titles = [f"Movie {i}" for i in range(n_movies)]
    overviews = [f"Overview for movie {i}" for i in range(n_movies)]

    # Create random genres
    all_genres = ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"]
    genres_data = []
    for i in range(n_movies):
        n_genres = np.random.randint(1, 4)
        movie_genres = np.random.choice(all_genres, n_genres, replace=False)
        genres_json = [{"name": genre} for genre in movie_genres]
        genres_data.append(str(genres_json))

    movies_df = pd.DataFrame({
        'original_title': titles,
        'overview': overviews,
        'genres': genres_data,
        'vote_average': np.random.uniform(5, 10, n_movies),
        'vote_count': np.random.randint(100, 10000, n_movies)
    })

    # Process genres
    movies_df['genres_list'] = movies_df['genres'].apply(
        lambda x: [i['name'] for i in ast.literal_eval(x)]
    )

    return movies_df

def create_mock_similarity_matrix(movies_df):
    """Create a mock similarity matrix"""
    n_movies = len(movies_df)
    # Create random similarity matrix
    sim_matrix = np.random.rand(n_movies, n_movies)

    # Make it symmetric
    sim_matrix = (sim_matrix + sim_matrix.T) / 2

    # Set diagonal to 1 (self-similarity)
    np.fill_diagonal(sim_matrix, 1.0)

    # Ensure values are between 0 and 1
    sim_matrix = np.clip(sim_matrix, 0, 1)

    return sim_matrix

def assert_valid_similarity_matrix(sim_matrix):
    """Assert that a similarity matrix has valid properties"""
    # Check shape is square
    assert sim_matrix.shape[0] == sim_matrix.shape[1]

    # Check diagonal is 1
    np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(sim_matrix.shape[0]))

    # Check symmetry
    assert np.allclose(sim_matrix, sim_matrix.T)

    # Check values between 0 and 1
    assert np.all((sim_matrix >= 0) & (sim_matrix <= 1))

def calculate_genre_overlap_test(rec_titles, input_genres, movies_df, indices):
    """Test version of genre overlap calculation"""
    rec_genres = set()
    for rec in rec_titles:
        if rec in indices.index:
            rec_idx = indices[rec]
            rec_genres.update(movies_df.iloc[rec_idx]['genres_list'])

    overlap = len(input_genres.intersection(rec_genres))
    return overlap / len(input_genres) if input_genres else 0

def time_function(func, *args, **kwargs):
    """Time a function execution"""
    import time
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time

# Test data fixtures
@pytest.fixture
def sample_movies():
    """Fixture for sample movie data"""
    return create_mock_movie_data(5)

@pytest.fixture
def sample_similarity_matrix(sample_movies):
    """Fixture for sample similarity matrix"""
    return create_mock_similarity_matrix(sample_movies)