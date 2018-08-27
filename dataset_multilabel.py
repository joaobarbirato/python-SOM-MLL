from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from kohonen import Mapa
from kfold import train_test_kfold
from metrics import *
import csv
from time import time, clock
from datetime import date

LISTA_N_EXEMPLOS    =   [150]
LISTA_N_ATRIBUTOS   =   [  5,  15,  30]
LISTA_N_CLASSES     =   [  3,   5,  10]
LISTA_N_DIMENSOES   =   [ 10,  20,  25]
LISTA_INTERVALOS    =   [(-1,1), (0,1)]
LISTA_N_VIZINHOS    =   [  1,   2,   3]

soma_p_base = []
soma_c_base = []
soma_f_base = []
soma_p_prop = []
soma_c_prop = []
soma_f_prop = []

timestamp = time()
with open('resultados/make_multilabel/' + str(date.today()) + '_' + str(timestamp) + '_' + '_resultado.csv', 'w') as tabela:
    writer = csv.DictWriter(tabela, fieldnames=['metodo','n_dimensoes', 'n_classes', 'n_atributos', 'n_exemplos','thr','precisao','cobertura','medida_f'])
    writer.writeheader()
    t0 = clock()
    for n_exemplos in LISTA_N_EXEMPLOS:
        for n_atributos in LISTA_N_ATRIBUTOS:
            for n_classes in LISTA_N_CLASSES:
                x, y = make_multilabel_classification(n_samples=n_exemplos,n_features=n_atributos, n_classes=n_classes)
                x = scale(x)
                for n_dimensoes in LISTA_N_DIMENSOES:
                    som = Mapa(taxa=.1,dimensao=n_dimensoes,fi0=8,features=n_atributos,decaimento=10000, nclasses=n_classes)
                    for bool_val in [False, True]:
                        for intervalo in LISTA_INTERVALOS:
                            thr = (intervalo[1]+intervalo[0])/2
                            [y_v,y_p] = train_test_kfold(10,True,None,som,x,y,vizinhos=bool_val, thr=0, n_vizinhos=1, intervalo=intervalo)
                            row = {
                                'metodo': "PROPOSTO" if bool_val else "BASELINE",
                                'n_dimensoes': n_dimensoes,
                                'n_classes': n_classes,
                                'n_atributos': n_atributos,
                                'n_exemplos': n_exemplos,
                                'thr': thr,
                                'precisao': precision(y_v, y_p),
                                'cobertura': recall(y_v, y_p),
                                'medida_f': medida_f(y_v, y_p),
                            }
                            if bool_val:
                                soma_p_prop.append(row['precisao'])
                                soma_c_prop.append(row['cobertura'])
                                soma_f_prop.append(row['medida_f'])
                            else:
                                soma_p_base.append(row['precisao'])
                                soma_c_base.append(row['cobertura'])
                                soma_f_base.append(row['medida_f'])
                            writer.writerow(row)
                            print(row)

print("Finalizando overall de make_multilabel...")
with open('resultados/make_multilabel/' + str(date.today()) + '_' + str(timestamp) + '_' + '_overall.csv', 'w') as tabela:
    writer = csv.DictWriter(tabela, fieldnames=[
        'soma_p_prop',
        'soma_c_prop',
        'soma_f_prop',
        'soma_p_base',
        'soma_c_base',
        'soma_f_base'
        ])
    writer.writeheader()
    row = {
        'soma_p_prop': sum(soma_p_prop)/len(soma_p_prop),
        'soma_c_prop': sum(soma_c_prop)/len(soma_c_prop),
        'soma_f_prop': sum(soma_f_prop)/len(soma_f_prop),
        'soma_p_base': sum(soma_p_base)/len(soma_p_base),
        'soma_c_base': sum(soma_c_base)/len(soma_c_base),
        'soma_f_base': sum(soma_f_base)/len(soma_f_base),
    }
    writer.writerow(row)
    print(row)

print('Overall finalizado!')
print("CSV pronto! Apos "+ str(clock()-t0) + "Segundos!")
