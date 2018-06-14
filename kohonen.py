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
			soma = soma + v/len(vec)
		# for i in range(self.nclasses):
		# 	soma[i] = soma[i]/len(vec)

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
	def decision_function(self, entrada, k=1, thr=0.5, vizinhos=True):
		menor = np.linalg.norm(entrada - self.pesos[0][0])
		x = 0
		y = 0
		for i in range(self.dimensao):
			for j in range(self.dimensao):
				if(menor > np.linalg.norm(entrada - self.pesos[i][j])):
					menor = np.linalg.norm(entrada - self.pesos[i][j])
					x = i
					y = j

		if(vizinhos):
			vTotal = []
			for i in range(k): # camada de vizinhos
				for xi in range(x-k-1,x+k):
					if(xi >= 0):
						for yi in range(y-k-1,y+k):
							if(yi >= 0):
								for l in self.mapeadosNeuronios[xi,yi]:
									vTotal.append(self.classesMapeadas[l])

			vSM = []
			for i in range(len(vTotal)):
				vSM.append(self._somaEMedia(vTotal[i]))

			#out = preprocessing.scale(self._somaEMedia(vSM))#
			out = self._somaEMedia(vSM)
			for i in range(len(out)):
				if(out[i] >= thr):
					out[i] = 1
				else:
					out[i] = 0

			return out
		else:
			v = []
			for l in self.mapeadosNeuronios[x,y]:
				for cm in self.classesMapeadas[l]:
					v.append(cm)
				
			if(v == []):
				return None
			
			out = self._somaEMedia(v)
			for i in range(len(out)):
				if(out[i] >= thr):
					out[i] = 1
				else:
					out[i] = 0
			return out

	# retorna numero de instancias mapeadas
	def get_n_inst(self):
		return self.iMapeados