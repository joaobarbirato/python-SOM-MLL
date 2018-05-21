from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from kohonen import Mapa

x, y = make_multilabel_classification(n_samples=150,n_features=4)

x = scale(x)
som = Mapa(taxa=.1,dimensao=23,fi0=8,features=4,decaimento=10000, nclasses=5)
kf = KFold(n_splits=10, shuffle=True, random_state=None)
for iteracao in range(10):
    print("Iteracao ", iteracao+1,"!")
    for indices_treino, indices_teste in kf.split(x):
        som.fit(x[indices_treino], y[indices_treino])

print(som.decision_function([0, 0.2, -0.5, 0.8], vizinhos=False))
