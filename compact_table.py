import csv
import os
import operator

max_rows = 5

all_sem_v = []

all_com_v = []

with open(str(os.getcwd()) + '/resultados/2018-06-16_2_resultado.csv','r') as arq_leitura:
    reader = csv.reader(arq_leitura)
    header = next(reader, None)
    maiores = sorted(reader, key=operator.itemgetter(7), reverse=False)[-2*max_rows:][::-1]

with open(str(os.getcwd()) + '/resultados/2018-06-16_2_resultado_compactado.csv','w') as arq_escrita:
    writer = csv.writer(arq_escrita)
    writer.writerow(header)
    for row in maiores:
        writer.writerow(row)