# Simple Linear Regression Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset

dataset = pd.read_csv(r'C:\Users\Windows10 Pro\Downloads\DataScience_AI\2025\3. Jun2025\11062025\Salary_Data.csv')

# Split dataset into X,y

x = dataset.iloc[:,:-1]

y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression   # LinearRegression is Algorithm  & linear_model is ML Model 
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# How do you know Regressor model is accuracy or not
y_pred = regressor.predict(x_test)

# Comparision
comparision = pd.DataFrame({'Actual':y_test, 'Predicted': y_pred})
print(comparision)

bias =  regressor.score(x_train, y_train)
print(bias)

variance = regressor.score(x_test, y_test)
print(variance)

# Statstical 
dataset.mean()
dataset['Salary'].mean()

dataset.mean()
dataset['YearsExperience'].mean()

# Median
dataset['Salary'].median()
dataset['YearsExperience'].median()

# Mode
dataset['Salary'].mode()
dataset['YearsExperience'].mode()

# Variance
dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()

# Standard Deviation
dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()

# Coefficient Variance (CV)
from scipy.stats import variation
variation(dataset.values)

variation(dataset['Salary'])
variation(dataset['YearsExperience'])

# Correaltion
dataset.corr()
dataset['Salary'].corr(dataset['YearsExperience'])

# Sekewness
dataset.skew()
dataset['Salary'].skew()

# Standard Error
dataset.sem()
dataset['Salary'].sem()

# Z Score
import scipy.stats as stats
dataset.apply(stats.zscore)

# Degree of Freedom
a = dataset.shape[0]  # gives no.of rows
b = dataset.shape[1]  # gives no.of columns

degree_of_freedom = a-b
print(degree_of_freedom)

# Sum of square regresso (SSR)
y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print(SSR)

# SSE
y = y[0:6]
SSE = np.sum((y-y_pred)**2)
print(SSE)

# SST
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)

# R Square
r_square = 1- (SSR/SST)
print(r_square)





# Visualizing 
plt.scatter(x_test,y_test, color='red')   # Real Salary data
plt.plot(x_train, regressor.predict(x_train), color='blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Validations (Predict Future data)
m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

# Predict 12 years experience guy
y_12 = m_slope*12+c_intercept   # y=mx+c
print(y_12)



