import csv
import os


with open(str(os.getcwd()) + '/resultados/2018-06-16_2_resultado.csv','r') as arq_leitura:
    reader = csv.reader(arq_leitura)
    dados = {
                'nao_vizinhos': 0,
                'vizinhos': 0,
                'media_nao_vizinhos_p': 0,
                'media_nao_vizinhos_r': 0,
                'media_nao_vizinhos_f': 0,
                'media_vizinhos_p': 0,
                'media_vizinhos_r': 0,
                'media_vizinhos_f': 0,
            }
    i = 0
    for row in reader:
        if i!=0:
            if row[0] == 'False' :
                dados['nao_vizinhos'] += 1
                dados['media_nao_vizinhos_p'] += float(row[5])
                dados['media_nao_vizinhos_r'] += float(row[6])
                dados['media_nao_vizinhos_f'] += float(row[7])
            else:
                dados['vizinhos'] += 1
                dados['media_vizinhos_p'] += float(row[5])
                dados['media_vizinhos_r'] += float(row[6])
                dados['media_vizinhos_f'] += float(row[7])
        else:
            i = 1

dados['media_nao_vizinhos_p'] /= dados['nao_vizinhos']
dados['media_nao_vizinhos_r'] /= dados['nao_vizinhos']
dados['media_nao_vizinhos_f'] /= dados['nao_vizinhos']

dados['media_vizinhos_p'] /= dados['vizinhos']
dados['media_vizinhos_r'] /= dados['vizinhos']
dados['media_vizinhos_f'] /= dados['vizinhos']

print("medida_f media sem vizinhos: " + str(dados['media_nao_vizinhos_f']))
print("medida_f media com vizinhos: " + str(dados['media_vizinhos_f']))
