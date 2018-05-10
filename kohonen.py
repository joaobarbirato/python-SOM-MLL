import numpy as np
from sklearn import datasets, preprocessing
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from skmultilearn.adapt import mlknn
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from random import random, seed

seed()

# Classe do Mapa de kohonen especializada para o dataset Iris
class MapaIris(BaseEstimator, ClassifierMixin):

	# Parâmetros
	def __init__(self, taxa=.1, dimensao=8, fi0=8, features=4, decaimento=1000):
		self.taxa = taxa 			# taxa de aprendizado
		self.decaimento = decaimento
		self.dimensao = dimensao 	# dimensão (largura) da grade de neuronios
		self.features = features

		self.fi = fi0 				# funcao de aprendizado 
		self.fi0 = fi0				# funcao de aprendizado (valor inicial)

		# matriz que representa a classe que o neuronio matriz[i][j] classifica
		self.matriz = np.ndarray(shape=(self.dimensao, self.dimensao), dtype=float, buffer=np.array([[-1 for i in range(self.dimensao)] for j in range(self.dimensao)])) #[[-1 for i in range(self.dimensao)] for j in range(self.dimensao)])
		# Array de pesos de cada neuronio
		self.pesos = np.ndarray(shape=(self.dimensao, self.dimensao, self.features), dtype=float, buffer=np.array([[[random() for i in range(features)] for j in range(self.dimensao)] for k in range(self.dimensao)]))

		# Epocas de aprendizado
		self.epoca = 1

		# Decaimento

	# metodos que sao chamados ao completar uma epoca
	def _resetMatriz(self):
		self.matriz = np.array([[-1 for i in range(self.dimensao)] for j in range(self.dimensao)])

	def _atualizaTaxa(self):
		self.taxa = self.taxa * np.exp(-self.epoca/self.decaimento)

	def _atualizaFi(self):
		self.fi = self.fi * np.exp(-self.epoca/(self.decaimento/np.log(self.fi0)))

	# Funcao de vizinhanca
	def _vizinhanca(self, entrada, x, y):
		return np.exp(-np.linalg.norm(- self.pesos[x][y])/(2*self.fi*self.fi))

	# metodo que muda a epoca
	def _mudaEpoca(self):
		self._atualizaTaxa()
		self._atualizaFi()
		self.epoca += 1

	# Classificacao pelo neuronio vencedor utilizando distancia euclidiana
	def _vencedor(self, entrada):
		i = 0
		j = 0
		_xvencedor = 0
		_yvencedor = 0
		menor = np.linalg.norm(entrada - self.pesos[0][0])
		achou = False
		while(not achou and i < self.dimensao):
			while(not achou and j < self.dimensao):
				if(self.matriz[i][j] == -1):
					achou = True
					menor = np.linalg.norm(entrada - self.pesos[i][j])
					_xvencedor = i
					_yvencedor = j
				j = j+1
			i = i+1

		for i in range(self.dimensao):
			for j in range(self.dimensao):
				if(menor > np.linalg.norm(entrada - self.pesos[i][j])):
					if(self.matriz[i][j] == -1):
						menor = np.linalg.norm(entrada - self.pesos[i][j])
						_xvencedor = i
						_yvencedor = j
		
		return([_xvencedor,_yvencedor])

	def _knn(self, x, y, k):
		labels = []
		labels.append(self.matriz[x][y])
		for i in range(k): # camada de vizinhos
			M = self.matriz[x-k:x+k+1, y-k:y+k+1]
			(nrows,ncols) = M.shape
			for xi in range(nrows):
				for yi in range(ncols):
					if(M[xi][yi] != -1 and M[xi][yi] not in labels):
						labels.append(M[xi][yi])

		return(labels)

	# Treinamento dado um dataset de entrada
	def fit(self, dTreino, saidas=None):		
		self._resetMatriz()
		_posicao = 0
		for vTreino in dTreino:
			[x, y] = self._vencedor(vTreino)
			viz = self._vizinhanca(vTreino, x, y)
			# procura o menor dos pesos e atualiza pesos
			self.pesos[x][y] = self.pesos[x][y] + self.taxa* viz *(vTreino - self.pesos[x][y])
			
			if(saidas is not None):
				self.matriz[x][y] = saidas[_posicao]

			_posicao = _posicao + 1

		self._mudaEpoca()
		return(self)

	def decision_function(self, entrada, k=1):
		menor = np.linalg.norm(entrada - self.pesos[0][0])

		for i in range(self.dimensao):
			for j in range(self.dimensao):
				if(menor > np.linalg.norm(entrada - self.pesos[i][j])):
					menor = np.linalg.norm(entrada - self.pesos[i][j])
					_xvencedor = i
					_yvencedor = j
		
		return(self._knn(_xvencedor,_yvencedor,k=k))

	# acesso publico a matriz
	def getMatriz(self):
		return(self.matriz)
		
dados = datasets.load_iris()
X = preprocessing.scale(dados['data'])
Y = np.ndarray(shape=(150,3), dtype=int, buffer=np.array([[0 for i in range(3)] for j in range(150)]))
for i in range(len(Y)):
	if(dados['target'][i] == 0):
		Y[i] = np.array([0, 0, 1])
	elif(dados['target'][i] == 1):
		Y[i] = np.array([0, 1, 0])
	elif(dados['target'][i] == 2):
		Y[i] = np.array([1, 0, 0])

dado_setosa = X[0]
dado_versicolor = X[50]
dado_virginica = X[100]

mapa = MapaIris(taxa=.1,dimensao=13,fi0=8,features=4,decaimento=10000)

kf = KFold(n_splits=10, shuffle=True, random_state=None)
for iteracao in range(10):
	print("Iteracao ", iteracao+1,"!")
	for indices_treino, indices_teste in kf.split(X):
		mapa.fit(X[indices_treino], dados['target'][indices_treino])

print(mapa.decision_function(dado_setosa))
print(mapa.decision_function(dado_versicolor))
print(mapa.decision_function(dado_virginica))