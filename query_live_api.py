import requests

url = 'https://fastapi-ml-app-u6be.onrender.com/infer'

input_data = {
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education-num": 13,
        "marital-status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 2174,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

# Send a POST request
response = requests.post(url, json=input_data)
# response = requests.get('https://fastapi-ml-app-u6be.onrender.com', data=input_data)

# Print the response content
print(f'Returned text: {response.text}')
print(f'Returned Status Code: {response.status_code}')

