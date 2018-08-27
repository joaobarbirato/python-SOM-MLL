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
class Mapa(BaseEstimator, ClassifierMixin):

	# Parâmetros
	def __init__(self, taxa=.1, dimensao=8, fi0=8, features=4, decaimento=1000, nclasses=3):
		self.taxa = taxa 			# taxa de aprendizado
		self.decaimento = decaimento
		self.dimensao = dimensao 	# dimensão (largura) da grade de neuronios
		self.features = features

		self.fi = fi0 				# funcao de aprendizado 
		self.fi0 = fi0				# funcao de aprendizado (valor inicial)
		self.nclasses = nclasses
		# Dicionario cuja chave eh o neuronio e o valor eh a posicao
		self.mapeadosNeuronios = {} # dict[i,j] = [pos1, pos2]
		for i in range(self.dimensao):
			for j in range(self.dimensao):
				self.mapeadosNeuronios[i,j] = []
		# Numero de exemplos mapeados no treino (auxiliar no dicionario de classes mapeadas)
		self.iMapeados = 0
		# Dicionario cuja chave eh a posicao e o valor eh a classe
		self.classesMapeadas = {}
		# Array de pesos de cada neuronio
		self.pesos = np.ndarray(shape=(self.dimensao, self.dimensao, self.features), dtype=float, buffer=np.array([[[random() for i in range(features)] for j in range(self.dimensao)] for k in range(self.dimensao)]))

		# Epocas de aprendizado
		self.epoca = 1

		# Decaimento

	# metodos que sao chamados ao completar uma epoca
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

	def _somaEMedia(self, vec):
		soma = [0 for i in range(self.nclasses)]
		for v in vec:
			# print(v)
			soma = soma + v/len(vec)
		# for i in range(self.nclasses):
		# 	soma[i] = soma[i]/len(vec)
		# print(soma)
		return soma
	# Classificacao pelo neuronio vencedor utilizando distancia euclidiana
	def _vencedor(self, entrada):
		i = 0
		j = 0
		_xvencedor = 0
		_yvencedor = 0
		menor = np.linalg.norm(entrada - self.pesos[0][0])
		for i in range(self.dimensao):
			for j in range(self.dimensao):
				if(menor > np.linalg.norm(entrada - self.pesos[i][j])):
					menor = np.linalg.norm(entrada - self.pesos[i][j])
					_xvencedor = i
					_yvencedor = j

		return([_xvencedor,_yvencedor])

	# Treinamento dado um dataset de entrada
	# (nao-supervisionado)
	def fit(self, dTreino, dSaidaBin):		
		for vTreino in dTreino:
			[x, y] = self._vencedor(vTreino)
			viz = self._vizinhanca(vTreino, x, y)
			# procura o menor dos pesos e atualiza pesos
			self.pesos[x][y] = self.pesos[x][y] + self.taxa* viz *(vTreino - self.pesos[x][y])

			self.classesMapeadas[self.iMapeados] = dSaidaBin
			self.mapeadosNeuronios[x,y].append(self.iMapeados)
			self.iMapeados += 1

		self._mudaEpoca()
		return self

	# funcao de treino
	def decision_function(self, entradas, k=1, thr=0.5, vizinhos=True):
		p_matrix = [] if(len(entradas) > 1) else None
		for entrada in entradas:
			# Select winner neuron from neuron grid
			menor = np.linalg.norm(entrada - self.pesos[0][0])
			x = 0
			y = 0
			for i in range(self.dimensao):
				for j in range(self.dimensao):
					if(menor > np.linalg.norm(entrada - self.pesos[i][j])):
						menor = np.linalg.norm(entrada - self.pesos[i][j])
						x = i
						y = j
			
			
			if(vizinhos): # proposto
				vTotal = []
				for i in range(k): # camada de vizinhos
					for xi in range(x-k-1,x+k):
						if(xi >= 0):
							for yi in range(y-k-1,y+k):
								if(yi >= 0):
									for l in self.mapeadosNeuronios[xi,yi]: # Get training instances mapped to winner neuron
										vTotal.append(self.classesMapeadas[l])

				vSM = []
				for i in range(len(vTotal)):
					vSM.append(self._somaEMedia(vTotal[i]))

			else: # baseline
				vSM = []
				for l in self.mapeadosNeuronios[x,y]: # Get training instances mapped to winner neuron
					for cm in self.classesMapeadas[l]:
						vSM.append(cm) 
				if(vSM == []):
					return None
					
			vSM = self._somaEMedia(vSM) # Get prototype vector
			p_matrix.append(vSM) # Associate prototype to instance

		scaler = preprocessing.MinMaxScaler(copy=True, feature_range=(-1,1))
		p_matrix = scaler.transform(p_matrix)
		for i in range(len(p_matrix)):
			for j in range(len(p_matrix[0])):
				p_matrix[i][j] = 1 if p_matrix[i][j] >= thr else 0

		return p_matrix
	
	# retorna numero de instancias mapeadas
	def get_n_inst(self):
		return self.iMapeados