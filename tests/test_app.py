import sys
sys.path.append('./')
from fastapi.testclient import TestClient
from main import app, load_components
import pytest
import asyncio


@pytest.fixture(scope="module")
def test_app():
    asyncio.run(load_components())  # Ensure async startup event is awaited
    # If you need to perform any setup before starting the app, do it here
    client = TestClient(app)
    yield client  # this provides the test client for tests
    # If you need to perform any teardown after tests, do it here

def test_read_main(test_app):
    response = test_app.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, welcome to our API!"}

def test_inference_result_1(test_app):
    # Replace with appropriate input data for a specific inference result
    input_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Husband",
        "race":"White",
        "sex":"Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }
    response = test_app.post("/infer/", json=input_data)
    assert response.status_code == 200
    # Assert based on expected outcome for this input
    assert response.json() == {"Inference": "['<=50K']"}

def test_inference_result_2(test_app):
    # Another set of data leading to a different inference result
    input_data = {
        "age": 50,
        "workclass": "State-gov",
        "fnlgt": 337895,
        "education": "Prof-school",
        "education-num": 15,
        "marital-status": "Married-civ-spouse",
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 52,
        "native-country": "United-States"
    }
    response = test_app.post("/infer/", json=input_data)
    assert response.status_code == 200
    # Assert based on expected outcome for this input
    assert response.json() == {"Inference": "['>50K']"}