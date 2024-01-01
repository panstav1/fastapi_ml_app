from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, welcome to our API!"}

def test_inference_result_1():
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
    response = client.post("/infer/", json=input_data)
    assert response.status_code == 200
    # Assert based on expected outcome for this input
    assert response.json() == {"Inference": "['<=50K']"}

# def test_inference_result_2():
#     # Another set of data leading to a different inference result
#     input_data = {
#         "age": 45,
#         "workclass": "Self-emp",
#         # ... other fields
#     }
#     response = client.post("/infer/", json=input_data)
#     assert response.status_code == 200
#     # Assert based on expected outcome for this input
#     assert response.json() == {"expected": "different outcome"}
#
# # Add more test cases as needed for each unique outcome of your model
