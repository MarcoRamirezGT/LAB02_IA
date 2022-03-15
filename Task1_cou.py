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





indep = df2[['Store','Week','Temperature','Fuel_Price','Unemployment','Holiday_Flag','CPI']]
#indep = df2[['Week']]
dep= df2['Weekly_Sales']



## Selecciona filas aleatoriamente y las separa en entrenamiento y test


X_training , X_testing, Y_training , Y_testing = train_test_split(indep,dep, test_size=0.3)
X_training_df , X_testing_df = pd.DataFrame(X_training) , pd.DataFrame(X_testing)

#print(Y_training)
polynomial = PolynomialFeatures(degree=5)

#primero es necesario ordenar para tener grafica legible 

Y_training =Y_training.sort_index()
Y_testing = Y_testing.sort_index()

X_trainig_poly = polynomial.fit_transform(X_training_df.sort_index())
X_testing_poly = polynomial.fit_transform(X_testing_df.sort_index())


my_model = linear_model.LinearRegression()
my_model = my_model.fit(X_trainig_poly,Y_training )

my_coeficients = my_model.coef_
my_intercept = my_model.intercept_



generated_y = my_model.predict(X_testing_poly)

# graficamos la informacion del tes y la ifnormacion del test predicha
plt.scatter(X_testing['Week'], Y_testing, s=3)
plt.plot(X_testing_poly,generated_y,linewidth=1, color='g')
plt.show()


print('Precisión del modelo:\n')
print(my_model.score(X_trainig_poly, Y_training))
