import pickle
import warnings
warnings.filterwarnings("ignore")

#Machine learning libraries 
import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#reading the dataset
df = pd.read_csv("./dataset/Salary_Data.csv")

#splitting independent and dependent features and reshaping them 
X = df['YearsExperience'].values
y = df['Salary'].values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

#splitting training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#creating a Linear Regression model and fitting the training data to it
lin_regr = LinearRegression()
lin_regr.fit(X_train, y_train)

#making predicitons on the testing data
y_pred = lin_regr.predict(X_test)


pickle.dump(lin_regr,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

