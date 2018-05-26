from sklearn.model_selection import KFold
from kohonen import Mapa

def train_test_kfold(n_s, s, r_s, mapa, x, y):
    kf = KFold(n_splits=n_s, shuffle=s, random_state=r_s)
    y_previsto = []
    y_verdadeiro = []
    for iteracao in range(10):
        print("Iteracao ", iteracao+1,"!")
        for indices_treino, indices_teste in kf.split(x):
            mapa.fit(x[indices_treino], y[indices_treino])
            for i_teste in indices_teste:
                y_previsto.append(mapa.decision_function(x[i_teste]))
                y_verdadeiro.append(y[i_teste])

    return[y_verdadeiro,y_previsto]

#train_test_kfold(10,True,None,Mapa(taxa=.1,dimensao=23,fi0=8,features=4,decaimento=10000, nclasses=3),[],[])