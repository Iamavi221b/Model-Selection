## Importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_test(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    return X_train, X_test, y_train, y_test


def regression(data, **kwargs):
    X = data[kwargs['independent_col']]
    y = data[kwargs['dependent_col']]

    X_train, X_test, y_train, y_test = train_test(X, y)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    y_pred = regressor.predict(X_test)

    ## Visualising the Training set results
    plt.scatter(X_train, y_train, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary VS Experience (Training Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

    ## Visualising the Test set results
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Salary VS Experience (Test Set)')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()

def clustering(data, **kwargs):
    print(data, kwargs)


def model_selection(**kwargs):
    data = pd.read_csv('{}'.format(kwargs['file_name']))
    # print(data)

    if kwargs['model_type'] == 'regression':
        regression(data, **kwargs)
    elif kwargs['model_type'] == 'clustering':
        clustering(data, **kwargs)

model_selection(file_name='Salary_Data.csv', model_type='regression', 
independent_col = ['YearsExperience'], dependent_col = ['Salary']
)