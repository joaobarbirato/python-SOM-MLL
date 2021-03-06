from sklearn.model_selection import KFold
from datetime import date
from kohonen import Mapa
import os


def train_test_kfold(n_s, s, r_s, mapa, x, y, vizinhos, thr=0, n_vizinhos=1, intervalo=(-1,1)):
    kf = KFold(n_splits=n_s, shuffle=s, random_state=r_s)
    y_previsto = []
    y_verdadeiro = []
    iteracao = 1
    for indices_treino, indices_teste in kf.split(x):
        print("Iteracao ", iteracao, " !")
        mapa.fit(x[indices_treino], y[indices_treino])
        y_previsto_i = mapa.decision_function(
            entradas=x[indices_teste],
            k=n_vizinhos,
            thr=thr,
            vizinhos=False,
            intervalo_scale=intervalo
        )
        y_verdadeiro_i = y[indices_teste]
        if y_previsto_i is not None:
            [y_previsto.append(e) for e in y_previsto_i]
        if y_verdadeiro_i is not None:
            [y_verdadeiro.append(e) for e in y_verdadeiro_i]
        iteracao += 1
    return[y_verdadeiro,y_previsto]
