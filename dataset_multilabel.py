from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from kohonen import Mapa
from kfold import train_test_kfold
from metrics import *
import csv
from time import clock
from datetime import date

lista_n_exemplos    =   [150]#, 300, 450]
lista_n_atributos   =   [  5,  15,  30]
lista_n_classes     =   [  3,   5,  10]
lista_n_neuronios   =   [  15, 25,  50]


with open('resultados/'+str(date.today())+'_resultado.csv', 'w') as tabela:
    writer = csv.DictWriter(tabela, fieldnames=['usa_vizinhos','n_neuronios', 'n_classes', 'n_atributos', 'n_exemplos','precisao','cobertura','medida_f'])
    writer.writeheader()
    t0 = clock()
    for bool_val in [False, True]:
        for n_exemplos in lista_n_exemplos:
            for n_atributos in lista_n_atributos:
                for n_classes in lista_n_classes:
                    x, y = make_multilabel_classification(n_samples=n_exemplos,n_features=n_atributos, n_classes=n_classes)
                    x = scale(x)
                    for n_neuronios in lista_n_neuronios:
                        som = Mapa(taxa=.1,dimensao=23,fi0=8,features=n_atributos,decaimento=10000, nclasses=n_classes)
                        [y_v,y_p] = train_test_kfold(10,True,None,som,x,y,vizinhos=bool_val)
                        row = {   
                            'usa_vizinhos': bool_val,
                            'n_neuronios': n_neuronios,
                            'n_classes': n_classes,
                            'n_atributos': n_atributos,
                            'n_exemplos': n_exemplos,
                            'precisao': precision(y_v, y_p),
                            'cobertura': recall(y_v, y_p),
                            'medida_f': medida_f(y_v, y_p),
                        }
                        writer.writerow(row)
                        print(row)

print("CSV pronto! Apos "+ str(clock()-t0) + "Segundos!")