import pytest
from fastapi.testclient import TestClient
import io
from PIL import Image

@pytest.mark.api
class TestAPIEndpoints:
    """Test API endpoints"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "CropCare AI API is running!"
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    def test_predict_unauthorized(self, client):
        """Test prediction endpoint without authentication"""
        response = client.post("/predict")
        assert response.status_code == 403
    
    def test_predict_invalid_file(self, client, auth_headers):
        """Test prediction with invalid file"""
        response = client.post(
            "/predict",
            headers=auth_headers,
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        assert response.status_code == 400
        assert "File must be an image" in response.json()["detail"]
    
    def test_predict_valid_image(self, client, auth_headers):
        """Test prediction with valid image"""
        # Create a test image
        img = Image.new('RGB', (100, 100), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_bytes.seek(0)
        
        response = client.post(
            "/predict",
            headers=auth_headers,
            files={"file": ("test.jpg", img_bytes.getvalue(), "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "disease" in data
        assert "confidence" in data
        assert "treatment_advice" in data
        assert "image_id" in data
        assert 0 <= data["confidence"] <= 1
    
    def test_get_predictions_unauthorized(self, client):
        """Test getting predictions without authentication"""
        response = client.get("/predictions")
        assert response.status_code == 403
    
    def test_get_predictions_authorized(self, client, auth_headers):
        """Test getting predictions with authentication"""
        response = client.get("/predictions", headers=auth_headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_cors_headers(self, client):
        """Test CORS headers"""
        response = client.options("/")
        assert response.status_code == 200
        # CORS headers should be present
        assert "access-control-allow-origin" in response.headers
