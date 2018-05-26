from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from kohonen import Mapa
from kfold import train_test_kfold
from metrics import *

x, y = make_multilabel_classification(n_samples=150,n_features=4)
x = scale(x)

som = Mapa(taxa=.1,dimensao=23,fi0=8,features=4,decaimento=10000, nclasses=5)
[y_v,y_p] = train_test_kfold(10,True,None,som,x,y)

print(precision(y_v, y_p))
print(recall(y_v, y_p))
print(medida_f(y_v, y_p))