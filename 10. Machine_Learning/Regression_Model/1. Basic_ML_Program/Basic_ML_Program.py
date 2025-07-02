# 1. Import Libraries

import numpy as np   # ctrl + i

import matplotlib.pyplot as plt   # ctrl+i
 
import pandas as pd    # ctrl+i

# 2. Import Dataset

dataset = pd.read_csv(r"C:\Users\Windows10 Pro\Downloads\DataScience_AI\2025\3. Jun2025\09062025\Data.csv")
print(dataset)


dataset.isnull().sum()
# Purchase is dependent variable


# 3. Split data to X, y
# dataset is split into X (Independent Variable) & y (dependent variable)
X = dataset.iloc[:, :-1].values  # iloc is location

y = dataset.iloc[:,3].values

# 4. Transforming (categorical to Numerical)
# Fill missing values
from sklearn.impute import SimpleImputer   # default MEAN Strategy is called as System Parametr Tuning

imputer = SimpleImputer()

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

'''

# with MEDIAN
from sklearn.impute import SimpleImputer   # default MEAN Strategy is called as Hyperparameter Tuning

imputer = SimpleImputer(strategy='median')

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

# with MODE
from sklearn.impute import SimpleImputer   # default MEAN Strategy is called as Hyperparameter Tuning

imputer = SimpleImputer(strategy='most_frequent')

imputer = imputer.fit(X[:,1:3])

X[:,1:3] = imputer.transform(X[:,1:3])

'''
# Converts Categorical to Numerical
# X

from sklearn.preprocessing import LabelEncoder

labelencoder_X = LabelEncoder()

labelencoder_X.fit_transform(X[:,0])  # I want to converts categorical data into Numerical (Bangalore, Hyderabad etc.,)

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
 
# y

labelencoder_y = LabelEncoder()

y = labelencoder_X.fit_transform(y)


# train & test the dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2)

# if you run multiple times, X_train, X_test, y_train, y_test values will change then it will impact on Accuracy
# then we need to use random_state=0

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.8, test_size=0.2, random_state=0)














