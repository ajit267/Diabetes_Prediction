import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler, scale 
from sklearn.ensemble import RandomForestClassifier 

# Data Preparation
df = pd.read_csv('diabetes.csv')
df.columns = df.columns.str.lower()
df = df.rename(columns={'bloodpressure':'bp','diabetespedigreefunction':'dpf'})
df_copy = df.copy(deep=True)
df_copy[['glucose','bp','skinthickness','insulin','bmi']] = df_copy[['glucose','bp','skinthickness','insulin','bmi']].replace(0,np.NaN)
df_copy.isnull().sum()
df_copy['glucose'].fillna(df_copy['glucose'].mean(), inplace=True)
df_copy['bp'].fillna(df_copy['bp'].mean(), inplace=True)
df_copy['skinthickness'].fillna(df_copy['skinthickness'].median(), inplace=True)
df_copy['insulin'].fillna(df_copy['insulin'].median(), inplace=True)
df_copy['bmi'].fillna(df_copy['bmi'].median(), inplace=True)

df_copy.isnull().sum()


# Data split
x = df.drop(columns='outcome')
y = df['outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_train.shape,x_test.shape

sc = StandardScaler()
x_test = sc.fit_transform(x_test)
x_train = sc.fit_transform(x_train)


# Train the model
rf = RandomForestClassifier(n_estimators=20,random_state=0,n_jobs=-1)
rf.fit(x_train,y_train)


#save the model

import pickle
filename = 'diabetes_prediction.pkl'
with open(filename, 'wb') as f_out:
    pickle.dump(rf, f_out)

# Save the scaler for future use
scaler_filename = 'diabetes_scaler.pkl'
with open(scaler_filename, 'wb') as file:
    pickle.dump(scale, file)  

print(f'the o/p file is saved to {filename}')






