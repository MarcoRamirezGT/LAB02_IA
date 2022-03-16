
import numpy as np
from collections import Counter 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split




dataset = pd.read_csv('Oranges vs Grapefruit.csv')
df2 = pd.DataFrame(dataset)
print(df2)

X= df2[['diameter','weight','red','green','blue']]
y= df2['name']

X_training, X_testing, y_training, y_testing = train_test_split(X,y, test_size=0.3)



def calc_euclidean_distance(x1,x2):
	#Le calcula la raiz de la distancia pero pordria hacerse la comparacion de solo las distancias 
	return np.sqrt(np.sum((x1-x2)**2))



class knn:
	def __init__(self, k =2):
		self.k = knn

	def fit(self, X, y ):
		self.X_train = X
		self.y_train = y


	def predict(self, X):
		predicted_labels = [self.predict(x) for x in X]
		return np.array(predicted_labels)

	def _predict(self,x):
		#compute distnaces
		distances = [calc_euclidean_distance(x, x_train) for x_train in self.X_train]
		

		#get k nearest labels 

		k_indices = np.argsort(distances)[0:self.k]
		k_nearest_labels = [self.y_train[i] for i in  k_indices]

		#Solo nos interesa saber la etiqueta mas comun entre los k vecinos
		most_common = Counter(k_nearest_labels).most_common(1)

		#Regresamos la etiqueta del mas comun

		return most_common[0][0]


my_knn = knn(k=3)
my_knn.fit(X_training,y_training)
#predictions= my_knn.predict(X_test)


# graficar los datos de test y las predicciones

plt.scatter(X_testing['weight'], y_testing, s=3)
plt.show()