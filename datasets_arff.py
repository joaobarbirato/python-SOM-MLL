import csv
import arff
import numpy as np
from time import time
from metrics import *
from time import clock
from kohonen import Mapa
from datetime import date
from kfold import train_test_kfold
from sklearn.preprocessing import scale

DATASET_NUMEROCLASSES = (
    ('CAL500', 174),
    ('emotions', 6),
    ('flags', 12)
)
TAXA = .1                                           # TAXA de aprendizado
DECAIMENTO = 10000
N_DIMENSOES = 5

LISTA_INTERVALOS    =   [(-1,1)]
LISTA_N_VIZINHOS    =   [  1,   2,   3]

for dataset, n_classes in DATASET_NUMEROCLASSES:
    data = arff.load(open('datasets/'+dataset+'.arff','r'))
    exemplos = data['data']                             # dataset
    n_atributos = len(data['attributes']) - n_classes   # n. de atributos
    n_exemplos = len(exemplos)                          # n. de exemplos
    
    x = [data['data'][i][:-n_classes] for i in range(n_exemplos)]
    x = scale(x)
    aux_y = [[int(e) for e in (data['data'][i][-n_classes:])] for i in range(n_exemplos)]
    y = np.ndarray(shape=(n_exemplos, n_classes), dtype=int, buffer=np.array(aux_y, dtype=int))

    
    media_f_base = []
    media_f_prop_k1 = []
    media_f_prop_k2 = []
    media_f_prop_k3 = []
    
    timestamp = time()
    with open('resultados/' + dataset + '/' + str(date.today()) + '_' + str(timestamp) + dataset + '_' + '_resultado.csv', 'w') as tabela:
        writer = csv.DictWriter(tabela, fieldnames=['metodo','N_DIMENSOES','precisao','cobertura','medida_f'])
        writer.writeheader()
        t0 = clock()
        for bool_val in [False, True]:
            som = Mapa(taxa=TAXA,dimensao=N_DIMENSOES,fi0=8,features=n_atributos,decaimento=DECAIMENTO, nclasses=n_classes)
            for intervalo in LISTA_INTERVALOS:
                thr = (intervalo[1] + intervalo[0])/2
                if bool_val:
                    for n_vizinho in LISTA_N_VIZINHOS:
                        [y_v,y_p] = train_test_kfold(10,True,None,som,x,y,vizinhos=bool_val,thr=thr, n_vizinhos=n_vizinho, intervalo=intervalo)
                        row = {
                            'metodo': 'PROPOSTO' if bool_val else 'BASELINE',
                            'N_DIMENSOES': N_DIMENSOES,
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
                    [y_v,y_p] = train_test_kfold(10,True,None,som,x,y,vizinhos=bool_val,thr=thr,intervalo=intervalo)
                    row = {
                        'metodo': 'PROPOSTO' if bool_val else 'BASELINE',
                        'N_DIMENSOES': N_DIMENSOES,
                        'precisao': precision(y_v, y_p),
                        'cobertura': recall(y_v, y_p),
                        'medida_f': medida_f(y_v, y_p),
                    }
                    media_f_base.append(row['medida_f'])
                    writer.writerow(row)
                    print(row)

    print("Finalizando overall de " + dataset + "...")
    with open('resultados/' + dataset + '/' + str(date.today()) + '_' + str(timestamp) + dataset + '_' + '_overall.csv', 'w') as tabela:
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
    print("CSV de " + dataset + " pronto! Apos "+ str(clock()-t0) + "Segundos!")
