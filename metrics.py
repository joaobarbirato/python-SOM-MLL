#!venv/bin/python
def precision(y_vdd,y_prev):
    S = len(y_prev)
    T = len(y_prev[0])
    soma = 0
    for i in range(S):
        card_intersec = 0
        for j in range(T):
            if y_vdd[i][j] == y_prev[i][j] and y_vdd[i][j] == 1: # conta se sao iguais
                card_intersec += 1

        card_classes_prev = 0
        for j in range(T):
            if y_prev[i][j] == 1:
                card_classes_prev += 1

        soma += card_intersec / card_classes_prev
            
    return soma/S

def recall(y_vdd,y_prev):
    S = len(y_prev)
    T = len(y_prev[0])
    soma = 0
    for i in range(S):
        card_intersec = 0
        for j in range(T):
            if y_vdd[i][j] == y_prev[i][j] and y_vdd[i][j] == 1: # conta se sao iguais
                card_intersec += 1

        card_classes_vdd = 0
        for j in range(T):
            if y_vdd[i][j] == 1:
                card_classes_vdd += 1

        soma += card_intersec / card_classes_vdd
            
    return soma/S

def medida_f(y_vdd,y_prev):
    p = precision(y_vdd, y_prev)
    r = recall(y_vdd, y_prev)
    try:
        return (2*(p*r)/(p+r))
    except ZeroDivisionError:
        return 0