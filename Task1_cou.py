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
df2 = pd.DataFrame(dataset)

#convertimos la fecha a formato d/m/y
# agregamos una columna de numero de la semana

df2['Date'] = pd.to_datetime(df2['Date'] , dayfirst=True,)
df2['Week'] = df2['Date'].dt.week
#print(df2['Week'])
#print(df2)

## variable independiente la fecha, holiday, fuel_price, cpi, store, temperature
## variable dependiente : weekly sales
#Variable a predecir: weekly sales


# Variables independientes: Weekly_Sales, Unemployment, Fuel_Price, CPI
# Variables dependetientes: Date
# Variable a predecir Weekly_Sales
y = df2.iloc[:, 2].values
x = df2.iloc[:, 0:1].values


train_x, test_x, train_y, test_y = train_test_split(x,
                                                    y,
                                                    test_size=0.3)



## Selecciona filas aleatoriamente y las separa en entrenamiento y test

indep = df2[['Store','Week','Temperature','Fuel_Price','Unemployment','Holiday_Flag','CPI']]
#indep = df2[['Week','Temperature']]
dep= df2['Weekly_Sales']

X_training , X_testing, Y_training , Y_testing = train_test_split(indep,dep, test_size=0.3)
X_training_df , X_testing_df = pd.DataFrame(X_training) , pd.DataFrame(X_testing)

#print(Y_training)

polynomial = PolynomialFeatures(degree=5)

X_trainig_poly = polynomial.fit_transform(X_training_df)
X_testing_poly = polynomial.fit_transform(X_testing_df)


my_model = linear_model.LinearRegression()
my_model = my_model.fit(X_trainig_poly,Y_training )

my_coeficients = my_model.coef_
my_intercept = my_model.intercept_




#predicen resultados con el modelo hecho

#generated_x = np.linspace(1,50)
#generated_y = my_intercept + (generated_x*my_coeficients[1]) + (my_coeficients[2]*generated_x**2) + (my_coeficients[3]*generated_x**3)
#print(X_testing)
#print(Y_testing)


#primero es necesario ordenar

# sorted_indices = numpy.argsort(X_esting)
# sorted_X = X_train[sorted_indices]
# plt.plot(sorted_X, regressor.predict(sorted_X), color = 'blue')





print(X_testing_poly)
generated_y = my_model.predict(X_testing_poly)

# graficamos la informacion del tes y la ifnormacion del test predicha
plt.scatter(X_testing['Week'], Y_testing, s=3)
plt.plot(X_testing_poly,generated_y,linewidth=1, color='g')
plt.show()



print('Precisión del modelo:\n')
print(my_model.score(X_trainig_poly, Y_training))
