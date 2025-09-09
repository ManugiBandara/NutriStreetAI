import pytest
import os
import numpy as np
from app import app, preprocess_image, model

# ---------- Flask Test Client ---------- #
@pytest.fixture
def client():
    app.testing = True
    return app.test_client()

# ---------- Page Routes ---------- #

def test_landing_page(client):
    """Landing page loads"""
    res = client.get('/')
    assert res.status_code == 200
    assert b"Smart" in res.data or b"Food" in res.data

def test_home_page(client):
    """Home page loads and form is present"""
    res = client.get('/home')
    assert res.status_code == 200
    assert b'action="/classify"' in res.data
    assert b'name="disease"' in res.data
    assert b'name="food_image"' in res.data
    assert b'type="submit"' in res.data

def test_faq_page(client):
    """FAQ page loads"""
    res = client.get('/faq')
    assert res.status_code == 200

# ---------- Image Preprocessing ---------- #

def test_preprocess_image():
    """Image preprocessing returns correct shape"""
    test_path = "static/test_image.jpg"
    assert os.path.exists(test_path), "Missing test image: static/test_image.jpg"
    img = preprocess_image(test_path)
    assert isinstance(img, np.ndarray)
    assert img.shape == (1, 224, 224, 3)

def test_model_prediction():
    """Model returns valid predictions"""
    dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    prediction = model.predict(dummy_input)
    assert prediction.shape[1] > 0
    assert np.all(prediction >= 0)

# ---------- Classify Endpoint ---------- #

def test_classify_food(client):
    """Classify endpoint works with valid image"""
    test_path = "static/test_image.jpg"
    assert os.path.exists(test_path), "Missing test image: static/test_image.jpg"
    
    with open(test_path, "rb") as img:
        data = {
            "food_image": (img, "test_image.jpg"),
            "disease": "diabetes"
        }
        res = client.post("/classify", data=data, content_type="multipart/form-data")

    assert res.status_code == 200
    assert b"Food Name:" in res.data
    assert b"Suitability" in res.data

def test_classify_no_image(client):
    """Classification should fail if no image is uploaded"""
    data = {"disease": "diabetes"}
    res = client.post("/classify", data=data, content_type="multipart/form-data")
    assert res.status_code == 400
    assert b"Bad Request" in res.data  # fallback
