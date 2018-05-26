import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import KFold
from kohonen import Mapa
from kfold import train_test_kfold
from metrics import *

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

mapa = Mapa(taxa=.1,dimensao=23,fi0=8,features=4,decaimento=10000, nclasses=3)
[y_v,y_p] = train_test_kfold(10,True,None,mapa,X,Y)

print(precision(y_v, y_p))
print(recall(y_v, y_p))
print(medida_f(y_v, y_p))