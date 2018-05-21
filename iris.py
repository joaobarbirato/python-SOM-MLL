import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import KFold
from kohonen import Mapa

dados = datasets.load_iris()
X = preprocessing.scale(dados['data'])
Y = np.ndarray(shape=(150,3), dtype=int, buffer=np.array([[0,0,0] for i in range(150)]))
for i in range(len(dados['data'])):
	if(dados['target'][i] == 0):
		Y[i] = np.array([1,0,0])
	if(dados['target'][i] == 1):
		Y[i] = np.array([0,1,0])
	if(dados['target'][i] == 2):
		Y[i] = np.array([0,0,1])

dado_setosa = X[0]
dado_versicolor = X[50]
dado_virginica = X[100]

mapa = Mapa(taxa=.1,dimensao=23,fi0=8,features=4,decaimento=10000, nclasses=3)


kf = KFold(n_splits=10, shuffle=True, random_state=None)
for iteracao in range(10):
	print("Iteracao ", iteracao+1,"!")
	for indices_treino, indices_teste in kf.split(X):
		mapa.fit(X[indices_treino], Y[indices_treino])

print(mapa.decision_function(dado_setosa,thr=0))
print(mapa.decision_function(dado_versicolor,thr=0))
print(mapa.decision_function(dado_virginica,thr=0))