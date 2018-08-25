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
    ('mediamill', 101)
)

for teste in range(3):
    for dataset, n_classes in DATASET_NUMEROCLASSES:
        data = arff.load(open('datasets/'+dataset+'.arff','r'))
        exemplos = data['data']                             # dataset
        n_atributos = len(data['attributes']) - n_classes   # n. de atributos
        n_exemplos = len(exemplos)                          # n. de exemplos
        taxa = .1                                           # taxa de aprendizado
        decaimento = 10000
        lista_n_dimensoes = [ 10,  20,  25]
        x = scale(np.ndarray([data['data'][i][:-n_classes] for i in range(len(n_exemplos))]))
        y = np.ndarray([data['data'][i][-n_classes:] for i in range(len(n_exemplos))])

        soma_p_base = []
        soma_c_base = []
        soma_f_base = []
        soma_p_prop = []
        soma_c_prop = []
        soma_f_prop = []
        
        with open('resultados/' + dataset + '/' + str(date.today()) + str(time()) + dataset + '_' + str(teste) + '_resultado.csv', 'w') as tabela:
            writer = csv.DictWriter(tabela, fieldnames=['usa_vizinhos','n_dimensoes','precisao','cobertura','medida_f'])
            writer.writeheader()
            t0 = clock()
            for bool_val in [False, True]:
                
                for n_dimensoes in lista_n_dimensoes:
                    som = Mapa(taxa=taxa,dimensao=n_dimensoes,fi0=8,features=n_atributos,decaimento=decaimento, nclasses=n_classes)
                    [y_v,y_p] = train_test_kfold(10,True,None,som,x,y,vizinhos=bool_val)
                    row = {
                        'usa_vizinhos': bool_val,prop
                        'n_dimensoes': n_dimensoes,
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
        with open('resultados/' + dataset + '/' + str(date.today()) + str(time()) + dataset + '_' + str(teste) + '_overall.csv', 'w') as tabela:
            writer = csv.DictWriter(tabela, fieldnames=['usa_vizinhos','n_dimensoes','precisao','cobertura','medida_f'])
            writer.writeheader()
            row = {
                'soma_p_prop': sum(soma_p_prop)/len(soma_p_prop)
                'soma_c_prop': sum(soma_c_prop)/len(soma_c_prop)
                'soma_f_prop': sum(soma_f_prop)/len(soma_f_prop)
                'soma_p_base': sum(soma_p_base)/len(soma_p_base)
                'soma_c_base': sum(soma_c_base)/len(soma_c_base)
                'soma_f_base': sum(soma_f_base)/len(soma_f_base)
            }
            writer.writeheader(row)
            print(row)
        print('Overall finalizado!')
        print("CSV de " + dataset + " pronto! Apos "+ str(clock()-t0) + "Segundos!")