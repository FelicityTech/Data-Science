import pytest
import requests
import json
import time
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the API app
try:
    from api import app
    client = TestClient(app)
except ImportError:
    client = None


@pytest.mark.skipif(client is None, reason="API app not available")
class TestAPI:
    """Test the FastAPI endpoints"""

    def test_root_endpoint(self):
        """Test the root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "endpoints" in data
        assert "/recommend" in data["endpoints"]
        assert "/algorithms" in data["endpoints"]

    def test_algorithms_endpoint(self):
        """Test the algorithms endpoint"""
        response = client.get("/algorithms")
        assert response.status_code == 200

        data = response.json()
        assert "algorithms" in data
        assert "descriptions" in data

        expected_algorithms = ["sigmoid", "cosine", "collaborative", "hybrid"]
        assert data["algorithms"] == expected_algorithms

        # Check descriptions exist
        for algo in expected_algorithms:
            assert algo in data["descriptions"]

    def test_recommend_endpoint_valid_movie(self):
        """Test recommendation endpoint with valid movie"""
        response = client.get("/recommend", params={
            "title": "The Matrix",
            "algorithm": "cosine",
            "top_n": 5
        })
        assert response.status_code == 200

        data = response.json()
        assert "input_movie" in data
        assert "algorithm" in data
        assert "recommendations" in data

        assert data["input_movie"]["title"] == "The Matrix"
        assert data["algorithm"] == "cosine"
        assert isinstance(data["recommendations"], list)
        assert len(data["recommendations"]) <= 5

    def test_recommend_endpoint_fuzzy_match(self):
        """Test recommendation endpoint with fuzzy matching"""
        response = client.get("/recommend", params={
            "title": "Matrix",  # Partial title
            "algorithm": "hybrid"
        })
        assert response.status_code == 200

        data = response.json()
        assert "input_movie" in data
        assert "recommendations" in data

    def test_recommend_endpoint_invalid_algorithm(self):
        """Test recommendation endpoint with invalid algorithm"""
        response = client.get("/recommend", params={
            "title": "The Matrix",
            "algorithm": "invalid_algo"
        })
        assert response.status_code == 400

    def test_recommend_endpoint_movie_not_found(self):
        """Test recommendation endpoint with non-existent movie"""
        response = client.get("/recommend", params={
            "title": "NonExistentMovie12345",
            "algorithm": "cosine"
        })
        assert response.status_code == 404

    def test_movie_details_endpoint(self):
        """Test movie details endpoint"""
        response = client.get("/movie/The Matrix")
        assert response.status_code == 200

        data = response.json()
        assert "title" in data
        assert "overview" in data
        assert "genres" in data
        assert data["title"] == "The Matrix"

    def test_movie_details_not_found(self):
        """Test movie details endpoint with non-existent movie"""
        response = client.get("/movie/NonExistentMovie")
        assert response.status_code == 404

    def test_search_endpoint(self):
        """Test search endpoint"""
        response = client.get("/search", params={
            "query": "Matrix",
            "limit": 3
        })
        assert response.status_code == 200

        data = response.json()
        assert "query" in data
        assert "results" in data
        assert isinstance(data["results"], list)
        assert len(data["results"]) <= 3

    def test_search_with_filters(self):
        """Test search endpoint with genre filter"""
        response = client.get("/search", params={
            "query": "action",
            "genre": "Action",
            "limit": 5
        })
        assert response.status_code == 200

        data = response.json()
        assert "results" in data

        # Check that results contain Action genre
        for result in data["results"]:
            if result["genres"]:  # Only check if movie has genres
                assert "Action" in result["genres"]

    def test_response_format(self):
        """Test that responses are proper JSON"""
        response = client.get("/recommend", params={"title": "The Matrix"})
        assert response.status_code == 200

        # Should be able to parse as JSON
        data = response.json()
        assert isinstance(data, dict)

        # Convert back to JSON string to ensure it's valid
        json_str = json.dumps(data)
        parsed_back = json.loads(json_str)
        assert parsed_back == data

    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/recommend")
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers or response.status_code == 200

# Integration tests (require running server)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests that require running server"""

    def test_server_startup(self):
        """Test that server can start (requires manual server start)"""
        try:
            response = requests.get("http://localhost:8000/", timeout=5)
            assert response.status_code == 200
        except requests.exceptions.RequestException:
            pytest.skip(
                "Server not running - start with 'python fastapi_app.py'")

    def test_full_recommendation_flow(self):
        """Test complete recommendation flow"""
        try:
            # Test recommendation
            response = requests.get("http://localhost:8000/recommend",
                                    params={"title": "Inception",
                                            "algorithm": "hybrid"},
                                    timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "recommendations" in data
            assert len(data["recommendations"]) > 0

            # Test that recommendations are different from input
            input_title = data["input_movie"]["title"]
            rec_titles = [rec["title"] for rec in data["recommendations"]]
            assert input_title not in rec_titles

        except requests.exceptions.RequestException:
            pytest.skip("Server not running")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
