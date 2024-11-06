import requests

url = 'http://localhost:9696/predict'

patient = {
    'pregnancies': 2,
    'glucose': 138,
    'bloodpressure': 62,
    'skinthickness': 35,
    'insulin': 0,
    'bmi': 33.6,
    'diabetespedigreefunction': 0.127,
    'age': 47
}

try:
    response = requests.post(url, json=patient)
    if response.status_code == 200:
        data = response.json()
        print(data) 
    else:
        print(f"Error: Received status code {response.status_code}")
        print("Response text:", response.text)

except requests.exceptions.JSONDecodeError:
    print("Response is not in JSON format.")
    print("Response text:", response.text)
except requests.exceptions.RequestException as e:
    print("An error occurred while making the request:", e)