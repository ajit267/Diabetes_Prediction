## Look for Deployed project at
https://diabetespredictml.streamlit.app/

# Diabetes Prediction

This project is a machine learning model that predicts the likelihood of diabetes based on certain health metrics. The model is trained using a dataset containing features like blood pressure, BMI, and glucose levels, and aims to help in early detection of diabetes, allowing for timely intervention.

## About the Project

Diabetes is a chronic illness that affects millions globally, leading to significant health complications if untreated. This project applies a machine learning approach to predict the likelihood of diabetes based on various health metrics, providing a tool that can assist healthcare professionals and individuals in identifying potential risks.

## Dataset

The model is trained on a dataset that includes the following features:
1. **Pregnancies** - Number of times the patient has been pregnant.
2. **Glucose** - Plasma glucose concentration.
3. **Blood Pressure** - Diastolic blood pressure in mm Hg.
4. **Skin Thickness** - Triceps skinfold thickness in mm.
5. **Insulin** - 2-Hour serum insulin in mu U/ml.
6. **BMI** - Body mass index.
7. **Diabetes Pedigree Function** - A function which scores likelihood of diabetes based on family history.
8. **Age** - Age of the patient in years.

The target variable is:
- **Outcome** - 1 if the patient has diabetes, 0 otherwise.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/ajit267/Diabetes_Prediction.git

## Features
Data Preprocessing: Handling missing values, scaling, and normalizing the dataset.
Model Training: Various machine learning models are tested, including Logistic Regression, K-Nearest Neighbors, Decision Trees, and others.
Evaluation: The model is evaluated on accuracy, precision, recall, and F1 score.
Prediction: The trained model can predict the likelihood of diabetes for new inputs.
Model
The project uses several machine learning models to predict diabetes likelihood, including:

Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Each model's performance is evaluated, and the best-performing model is selected for predictions.

## Prerequisites
Python 3.12 or later
pip and pipenv for package management
Docker (optional, for containerization)
Setup
Clone the repository or download the project files.

## Navigate to the project directory:
Insert Code
Edit
Copy code
cd diabetes
Install dependencies using pipenv:

Insert Code
Edit
Copy code
pipenv install
Activate the virtual environment:

Insert Code
Edit
Copy code
pipenv shell

## Project Structure
diabetes_prediction.py: Script to train the model and save it
diabetes_predict.py: Flask application for serving predictions
predict.py: Client script to test the prediction service
Pipfile and Pipfile.lock: Dependency management files
Dockerfile: For containerizing the application
Training the Model
Ensure you have the diabetes.csv dataset in the project directory.

Run the training script:

Insert Code
Edit
Copy code
python diabetes_prediction.py
This will create two pickle files:

diabetes_prediction.pkl: The trained Random Forest model
diabetes_scaler.pkl: The StandardScaler for preprocessing
Running the Prediction Service
Start the Flask application:

Insert Code
Edit
Copy code
python diabetes_predict.py
The service will run on http://localhost:9696

Testing the Prediction Service
With the Flask app running, open a new terminal and run:

Insert Code
Edit
Copy code
python predict.py
This script sends a sample patient data to the prediction service and prints the result.

## Using Docker (Optional)
Build the Docker image:

Insert Code
Edit
Copy code
docker build -t diabetes-prediction .
Run the container:

Insert Code
Edit
Copy code
docker run -p 9696:9696 diabetes-prediction
Making Predictions
To make a prediction, send a POST request to http://localhost:9696/predict with JSON data containing patient information:

json
Insert Code
Edit
Copy code
{
    "pregnancies": 2,
    "glucose": 138,
    "bloodpressure": 62,
    "skinthickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetespedigreefunction": 0.127,
    "age": 47
}
The service will respond with a prediction indicating whether the patient is likely to have diabetes or not.

## Results
The models were evaluated on various metrics, with accuracy, precision, recall, and F1 score calculated for each model. The final model selection is based on the one with the best overall performance.

## Contributing
Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request. Make sure to include relevant tests for any new functionality.

## License
Distributed under the MIT License. See LICENSE for more information.

## Contact
Project Link: Diabetes Prediction on GitHub
