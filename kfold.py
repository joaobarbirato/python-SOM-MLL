from sklearn.model_selection import KFold
from kohonen import Mapa
import os


def train_test_kfold(n_s, s, r_s, mapa, x, y, vizinhos):
    kf = KFold(n_splits=n_s, shuffle=s, random_state=r_s)
    y_previsto = []
    y_verdadeiro = []
    for iteracao in range(10):
        print("Iteracao ", iteracao+1,"!")
        for indices_treino, indices_teste in kf.split(x):
            mapa.fit(x[indices_treino], y[indices_treino])
            for i_teste in indices_teste:
                y_previsto.append(mapa.decision_function(x[i_teste],vizinhos))
                y_verdadeiro.append(y[i_teste])
    return[y_verdadeiro,y_previsto]
