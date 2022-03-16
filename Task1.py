import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

# Cargamos el conjunto de datos
dataset = pd.read_csv('Walmart.csv')

dataset = pd.DataFrame(dataset)

# Variables independientes: Weekly_Sales, Unemployment, Fuel_Price, CPI
# Variables dependetientes: Date
# Variable a predecir Weekly_Sales


dataset['Date'] = pd.to_datetime(dataset['Date'], dayfirst=True,)
dataset['Week'] = dataset['Date'].dt.week
y = dataset.iloc[:, 2].values
x = dataset.iloc[:, 7:8].values

print(dataset)

train_x, test_x, train_y, test_y = train_test_split(x,
                                                    y,
                                                    test_size=0.3)

print(test_y)
print(train_y)

pol = PolynomialFeatures(degree=20)

# Se transforma las caracteristicas
x_train_poli = pol.fit_transform(train_x)
x_test_poli = pol.fit_transform(test_x)

pr = linear_model.LinearRegression()
pr.fit(x_train_poli, train_y)
Y_pred_pr = pr.predict(x_test_poli)

plt.scatter(test_x, test_y)
plt.plot(test_x, Y_pred_pr, color='red', linewidth=3)
plt.show()

print('DATOS DEL MODELO REGRESIÓN POLINOMIAL\n')

print('Valor de la pendiente o coeficiente "a":\n')
print(pr.coef_)
print('Valor de la intersección o coeficiente "b":\n')
print(pr.intercept_)
print('Precisión del modelo:\n')
print(pr.score(x_train_poli, train_y))
