import pytest
import pandas as pd
import numpy as np
import joblib
from fuzzywuzzy import process
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock data for testing
@pytest.fixture
def mock_data():
    """Create mock movie data for testing"""
    movies_data = {
        'original_title': ['The Matrix', 'Inception', 'Interstellar', 'The Dark Knight', 'Pulp Fiction'],
        'overview': ['A hacker discovers reality', 'A thief enters dreams', 'Space exploration', 'Batman fights Joker', 'Non-linear crime story'],
        'genres': ['[{"name": "Action"}, {"name": "Sci-Fi"}]', '[{"name": "Action"}, {"name": "Thriller"}]', '[{"name": "Adventure"}, {"name": "Drama"}]', '[{"name": "Action"}, {"name": "Crime"}]', '[{"name": "Crime"}, {"name": "Drama"}]'],
        'keywords': ['[{"name": "hacker"}]', '[{"name": "dream"}]', '[{"name": "space"}]', '[{"name": "batman"}]', '[{"name": "crime"}]']
    }

    movies_df = pd.DataFrame(movies_data)
    movies_df['genres_list'] = movies_df['genres'].apply(lambda x: [i['name'] for i in eval(x)])
    movies_df['keywords_list'] = movies_df['keywords'].apply(lambda x: [i['name'] for i in eval(x)])

    # Create mock similarity matrices
    n_movies = len(movies_df)
    mock_sim = np.random.rand(n_movies, n_movies)
    np.fill_diagonal(mock_sim, 1.0)  # Self-similarity should be 1

    indices = pd.Series(movies_df.index, index=movies_df['original_title']).drop_duplicates()

    return movies_df, mock_sim, indices

@pytest.fixture
def mock_models(mock_data):
    """Create mock models dictionary"""
    movies_df, mock_sim, indices = mock_data
    return {
        'sig': mock_sim,
        'cos': mock_sim,
        'collab_sim': mock_sim,
        'hybrid_sim': mock_sim,
        'indices': indices
    }

def test_indices_creation(mock_data):
    """Test that indices are created correctly"""
    movies_df, _, indices = mock_data

    assert len(indices) == len(movies_df)
    assert indices['The Matrix'] == 0
    assert 'Nonexistent Movie' not in indices

def test_fuzzy_matching(mock_data):
    """Test fuzzy matching functionality"""
    movies_df, _, indices = mock_data

    # Test exact match
    title = 'The Matrix'
    assert title in indices.index

    # Test fuzzy match
    closest, score = process.extractOne('Matrix', indices.index)
    assert closest == 'The Matrix'
    assert score == 100

    # Test partial match
    closest, score = process.extractOne('Incept', indices.index)
    assert closest == 'Inception'
    assert score > 80

def test_recommendation_function(mock_data):
    """Test the give_recommendations function"""
    movies_df, mock_sim, indices = mock_data

    # Mock the give_recommendations function
    def give_recommendations(title, sim_matrix, top_n=10):
        if title not in indices.index:
            closest, score = process.extractOne(title, indices.index)
            if score > 80:
                title = closest
            else:
                return f"Movie not found. Closest match: '{closest}' (score: {score})"

        idx = indices[title]
        sig_scores = list(enumerate(sim_matrix[idx]))
        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
        sig_scores = sig_scores[1:top_n+1]  # Skip self
        movie_indices = [i[0] for i in sig_scores]
        return movies_df['original_title'].iloc[movie_indices].tolist()

    # Test with exact title
    recs = give_recommendations('The Matrix', mock_sim, top_n=3)
    assert isinstance(recs, list)
    assert len(recs) == 3
    assert 'The Matrix' not in recs  # Should not recommend itself

    # Test with fuzzy match
    recs = give_recommendations('Matrix', mock_sim, top_n=2)
    assert isinstance(recs, list)
    assert len(recs) == 2

def test_similarity_matrix_properties(mock_data):
    """Test that similarity matrices have correct properties"""
    movies_df, mock_sim, indices = mock_data

    # Check shape
    assert mock_sim.shape == (len(movies_df), len(movies_df))

    # Check diagonal is 1 (self-similarity)
    np.testing.assert_array_equal(np.diag(mock_sim), np.ones(len(movies_df)))

    # Check symmetry
    assert np.allclose(mock_sim, mock_sim.T)

    # Check values are between 0 and 1
    assert np.all((mock_sim >= 0) & (mock_sim <= 1))

def test_genre_processing(mock_data):
    """Test genre list processing"""
    movies_df, _, _ = mock_data

    # Check that genres_list is created correctly
    assert movies_df.loc[0, 'genres_list'] == ['Action', 'Sci-Fi']
    assert movies_df.loc[1, 'genres_list'] == ['Action', 'Thriller']

    # Check that all movies have genre lists
    assert all(isinstance(genres, list) for genres in movies_df['genres_list'])

def test_model_persistence(mock_models, tmp_path):
    """Test saving and loading models"""
    models = mock_models

    # Save models
    model_path = tmp_path / "test_models.pkl"
    joblib.dump(models, model_path)

    # Load models
    loaded_models = joblib.load(model_path)

    # Check that all keys are present
    expected_keys = ['sig', 'cos', 'collab_sim', 'hybrid_sim', 'indices']
    assert all(key in loaded_models for key in expected_keys)

    # Check that matrices have same shape
    for key in ['sig', 'cos', 'collab_sim', 'hybrid_sim']:
        np.testing.assert_array_equal(models[key].shape, loaded_models[key].shape)

    # Check that indices are identical
    pd.testing.assert_series_equal(models['indices'], loaded_models['indices'])

if __name__ == "__main__":
    pytest.main([__file__])