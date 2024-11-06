import pickle
import numpy as np # type: ignore
import pandas as pd # type: ignore
from flask import Flask # type: ignore
from flask import request # type: ignore
from flask import jsonify # type: ignore
from sklearn.preprocessing import StandardScaler

# Load the Random Forest model
model_filename = 'diabetes_prediction.pkl'
with open(model_filename, 'rb') as f_in:
    rf = pickle.load(f_in)

# Load the scaler
scaler_filename = 'diabetes_scaler.pkl'
with open(scaler_filename, 'rb') as f_in:
    scaler = pickle.load(f_in)


app = Flask('diabetes')

@app.route('/predict', methods=['POST'])
def predict():
    patient = request.get_json()
    X_new = np.array([list(patient.values())])
    y_pred_dv = rf.predict(X_new)

    if y_pred_dv[0] == 1:
        result = 'Oops! You have diabetes.'
    else:
        result = "Great! You don't have diabetes."

    return jsonify({"prediction": result})
 
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

