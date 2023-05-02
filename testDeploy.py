import requests
import json
import numpy as np

# the endpoint URL
endpoint = 'http://localhost:8080/v1/models/my-model/predict'

#  the sample input
data = {
    'inputs': {
        'input': [[5.1, 3.5, 1.4, 0.2]]
    }
}

# a POST request to the endpoint with the sample input
response = requests.post(endpoint, json=data)

# Parse the response
if response.status_code == 200:
    result = json.loads(response.text)
    output = np.array(result['outputs']['output'])
    print('Prediction:', output)
else:
    print('Error:', response.text)
