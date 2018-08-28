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
LISTA_N_DIMENSOES   =   [  5,  10,  25]
LISTA_INTERVALOS    =   [(-1,1)]
LISTA_N_VIZINHOS    =   [  1,   2,   3]

media_f_base = []
media_f_prop_k1 = []
media_f_prop_k2 = []
media_f_prop_k3 = []

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
                            if bool_val:
                                for n_vizinho in LISTA_N_VIZINHOS:
                                    [y_v,y_p] = train_test_kfold(10,True,None,som,x,y,vizinhos=bool_val, thr=thr, n_vizinhos=1, intervalo=intervalo)
                                    row = {
                                        'metodo': "PROPOSTO" if bool_val else "BASELINE",
                                        'n_dimensoes': n_dimensoes,
                                        'n_classes': n_classes,
                                        'n_atributos': n_atributos,
                                        'n_exemplos': n_exemplos,
                                        'precisao': precision(y_v, y_p),
                                        'cobertura': recall(y_v, y_p),
                                        'medida_f': medida_f(y_v, y_p),
                                    }
                                    if n_vizinho == LISTA_N_VIZINHOS[0]:
                                        media_f_prop_k1.append(row['medida_f'])
                                    elif n_vizinho == LISTA_N_VIZINHOS[1]:
                                        media_f_prop_k2.append(row['medida_f'])
                                    elif n_vizinho == LISTA_N_VIZINHOS[2]:
                                        media_f_prop_k3.append(row['medida_f'])
                                    writer.writerow(row)
                                    print(row)
                            else:
                                [y_v,y_p] = train_test_kfold(10,True,None,som,x,y,vizinhos=bool_val, thr=thr, intervalo=intervalo)
                                row = {
                                    'metodo': "PROPOSTO" if bool_val else "BASELINE",
                                    'n_dimensoes': n_dimensoes,
                                    'n_classes': n_classes,
                                    'n_atributos': n_atributos,
                                    'n_exemplos': n_exemplos,                                    
                                    'precisao': precision(y_v, y_p),
                                    'cobertura': recall(y_v, y_p),
                                    'medida_f': medida_f(y_v, y_p),
                                }
                                media_f_base.append(row['medida_f'])
                                writer.writerow(row)
                                print(row)

print("Finalizando overall de make_multilabel...")
with open('resultados/make_multilabel/' + str(date.today()) + '_' + str(timestamp) + '_' + '_overall.csv', 'w') as tabela:
    writer = csv.DictWriter(tabela, fieldnames=[
        'media_f_base',
        'media_f_prop_k1',
        'media_f_prop_k2',
        'media_f_prop_k3'
        ])
    writer.writeheader()
    row = {
        'media_f_base': sum(media_f_base)/len(media_f_base),
        'media_f_prop_k1': sum(media_f_prop_k1)/len(media_f_prop_k1),
        'media_f_prop_k2': sum(media_f_prop_k2)/len(media_f_prop_k2),
        'media_f_prop_k3': sum(media_f_prop_k3)/len(media_f_prop_k3)
    }
    writer.writerow(row)
    print(row)

print('Overall finalizado!')
print("CSV pronto! Apos "+ str(clock()-t0) + "Segundos!")
