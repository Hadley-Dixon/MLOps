import requests
import json

# Test data - a sample diamond with all required features
test_data = {
    "features": {
        "y": 5.73,
        "carat": 0.23,
        "x": 3.95,
        "z": 2.43,
        "clarity_SI2": 0,
        "clarity_I1": 0,
        "color_J": 0,
        "clarity_SI1": 1,
        "color_I": 0,
        "clarity_VVS2": 0,
        "depth": 61.5,
        "color_H": 0
    }
}

# Send POST request to the API
response = requests.post(
    "http://localhost:8000/predict",
    json=test_data
)

# Print the response
print("Status Code:", response.status_code)
print("Response:", response.json()) 