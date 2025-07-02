# SVR Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Windows10 Pro\Downloads\DataScience_AI\2025\3. Jun2025\25062025\emp_sal.csv")

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

#---------------------------------------------

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


#---------------------------------------------


# SVR Model Support Vector Regression
from sklearn.svm import SVR
#kernel{'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'} or callable, default='rbf'

svr_model = SVR(kernel="rbf", degree=5, gamma='scale')   # ctrl+i
svr_model.fit(X,y)

# Prediction
svr_model_pred = svr_model.predict([[6.5]])
svr_model_pred           # employee 6.5 got less salary than employee 6 & employee 7. So we are going to Hyper parameter 


svr_model = SVR(kernel="poly", degree=5, gamma='scale')   # ctrl+i
svr_model.fit(X,y)

# Prediction
svr_model_pred = svr_model.predict([[6.5]])
svr_model_pred 

# Hyper Parameter tuning
svr_model = SVR(kernel="poly", degree=5, gamma='scale', c=10.0)   # ctrl+i
svr_model.fit(X,y)

# Prediction
svr_model_pred = svr_model.predict([[6.5]])
svr_model_pred 


#---------------------------------------------



# KNN (K Nearest Neighbour Regression)

from sklearn.neighbors import KNeighborsRegressor     # ctrl+i
knn_model = KNeighborsRegressor()
knn_model.fit(X,y)

# Prediction
knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

# ----------------
# Hyper Parameter Tuning
knn_model = KNeighborsRegressor(n_neighbors=5, weights='distance', p=1)
knn_model.fit(X,y)

# Prediction
knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

#-------------------
knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X,y)

# Prediction
knn_model_pred = knn_model.predict([[6.5]])
print(knn_model_pred)

#---------------------------------------------
# Decision Tree ML Regression Model
 
from sklearn.tree import DecisionTreeRegressor   # ctrl+i
dt_model = DecisionTreeRegressor()
dt_model.fit(X,y)

# Prediction
dt_model_pred = dt_model.predict([[6.5]])
print(dt_model_pred)

# Hyper Parameter Tuning



#---------------------------------------------
# Random Forest # run minimum 3-4 times then you can observe model because its flutating

from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
rf_model.fit(X,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

# without fluating put random_state = 0
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(random_state=0)
rf_model.fit(X,y)

rf_model_pred = rf_model.predict([[6.5]])
print(rf_model_pred)

