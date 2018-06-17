def precision(y_vdd,y_prev):
    S = len(y_prev)
    T = len(y_prev[0])
    soma = 0
    for i in range(S):
        len(y_vdd[i])
        for j in range(len(y_vdd[i])):
            if y_vdd[i][j] == y_prev[i][j]:
                soma += 1
    return soma/(S*T)

def recall(y_vdd,y_prev):
    S = len(y_vdd)
    T = len(y_prev[0])
    soma = 0
    for i in range(S):
        for j in range(len(y_vdd[i])):
            if y_vdd[i][j] == y_prev[i][j]:
                soma += 1
    return soma/(S*T)

def medida_f(y_vdd,y_prev):
    p = precision(y_vdd, y_prev)
    r = recall(y_vdd, y_prev)
    try:
        return (2*(p*r)/(p+r))
    except ZeroDivisionError:
        return 0