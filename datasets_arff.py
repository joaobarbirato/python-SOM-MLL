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
    ('yeast', 14)
)
TAXA = .1                                           # TAXA de aprendizado
DECAIMENTO = 10000
N_DIMENSOES = 5

LISTA_INTERVALOS    =   [(-1,1), (0,1)]
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

    soma_p_base = []
    soma_c_base = []
    soma_f_base = []
    soma_p_prop = []
    soma_c_prop = []
    soma_f_prop = []
    
    timestamp = time()
    with open('resultados/' + dataset + '/' + str(date.today()) + '_' + str(timestamp) + dataset + '_' + '_resultado.csv', 'w') as tabela:
        writer = csv.DictWriter(tabela, fieldnames=['metodo','N_DIMENSOES','precisao','cobertura','medida_f'])
        writer.writeheader()
        t0 = clock()
        for bool_val in [False, True]:
            som = Mapa(taxa=TAXA,dimensao=N_DIMENSOES,fi0=8,features=n_atributos,decaimento=DECAIMENTO, nclasses=n_classes)
            for intervalo in LISTA_INTERVALOS:
                thr = (intervalo[1] + intervalo[0])/2
                for n_vizinho in LISTA_N_VIZINHOS:
                    [y_v,y_p] = train_test_kfold(10,True,None,som,x,y,vizinhos=bool_val,thr=0, n_vizinhos=1, intervalo=intervalo)
                    row = {
                        'metodo': 'PROPOSTO' if bool_val else 'BASELINE',
                        'N_DIMENSOES': N_DIMENSOES,
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

    print("Finalizando overall de " + dataset + "...")
    with open('resultados/' + dataset + '/' + str(date.today()) + '_' + str(timestamp) + dataset + '_' + '_overall.csv', 'w') as tabela:
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
    print("CSV de " + dataset + " pronto! Apos "+ str(clock()-t0) + "Segundos!")
