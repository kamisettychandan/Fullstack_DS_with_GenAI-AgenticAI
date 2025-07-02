# Polynomial Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Windows10 Pro\Downloads\DataScience_AI\2025\3. Jun2025\24062025\emp_sal.csv")

X = dataset.iloc[:, 1:2].values  # : is entire dataset, I need 1:2

y = dataset.iloc[:, 2].values

# we have less data, so spliting not required

# linear model  -- linear algor ( degree - 1)
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Linear Regression Visualization
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg.predict(X), color='blue')
plt.title("Linear Regression Model (Linear Regression")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# We can build Non-Linear Model because Linear model we didn't get Accuracy

# How it is not Accuracy
lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred

# Polynomial Model (Non-Linear Model-by default degree 2)
from sklearn.preprocessing import PolynomialFeatures   # ctrl + i
poly_reg = PolynomialFeatures() 
X_poly = poly_reg.fit_transform(X)   # default degree is 2 1*1, 2*2, 3*3...10*10

poly_reg.fit(X_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

X_poly
poly_reg
lin_reg_2

# Polynomial Visualizations
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title("Poly Model (Polynomial Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

# Prediction
poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

# as per plot not satisfying then we are going with Hyper Parameter Tuning (we are puting degree 3 in Polynomial Model)
'''from sklearn.preprocessing import PolynomialFeatures   # ctrl + i
poly_reg3 = PolynomialFeatures(degree=3) 
X_poly = poly_reg3.fit_transform(X)   # default degree is 2 1*1, 2*2, 3*3...10*10

poly_reg3.fit(X_poly, y)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(X_poly,y)

# with degree 4
from sklearn.preprocessing import PolynomialFeatures   # ctrl + i
poly_reg4 = PolynomialFeatures(degree=4) 
X_poly = poly_reg4.fit_transform(X)   # default degree is 2 1*1, 2*2, 3*3...10*10

poly_reg4.fit(X_poly, y)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(X_poly,y)
'''
from sklearn.preprocessing import PolynomialFeatures

poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly, y)

lin_reg_5 = LinearRegression()
lin_reg_5.fit(X_poly, y)

# Polynomial Regression Visualization
plt.scatter(X,y,color='red')
plt.plot(X, lin_reg_5.predict(poly_reg.fit_transform(X)), color='Blue')
plt.title("Polynomial Model (Polynomial Regression Algorithm")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
