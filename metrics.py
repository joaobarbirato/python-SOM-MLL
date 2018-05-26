def precision(y_vdd,y_prev):
    S = len(y_prev)
    soma = 0
    for i in range(S):
        if(y_vdd[i] == y_prev[i]).all():
            soma += 1
    return soma/S

def recall(y_vdd,y_prev):
    S = len(y_vdd)
    soma = 0
    for i in range(S):
        if(y_vdd[i] == y_prev[i]).all():
            soma += 1
    return soma/S

def medida_f(y_vdd,y_prev):
    p = precision(y_vdd, y_prev)
    r = recall(y_vdd, y_prev)
    return (2*(p*r)/(p+r))