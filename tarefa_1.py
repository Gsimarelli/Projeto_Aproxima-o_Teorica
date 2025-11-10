import numpy as np

def eliminacao_gauss(M, d):
    """
    Resolve sistema linear usando eliminação de Gauss
    conforme algoritmo da Seção 3 do PDF
    """
    n = len(M)
    # Triangularização
    for j in range(n - 1):
        if M[j][j] == 0:
            for k in range(j + 1, n):
                if M[k][j] != 0:
                    # Troca as linhas j e k
                    M[j], M[k] = M[k], M[j]
                    d[j], d[k] = d[k], d[j]
                    break
            if M[j][j] == 0:
                print("A matriz é singular")
                return None
        
        for i in range(j + 1, n):
            c = -M[i][j] / M[j][j]
            for o in range(j, n):
                M[i][o] += c * M[j][o]
            d[i] += c * d[j]
    
    # Substituição regressiva
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = d[i]
        for j in range(i + 1, n):
            x[i] -= M[i][j] * x[j]
        x[i] /= M[i][i]
    
    return x

def busca_binaria(xi, x_estrela):
    """
    Implementa algoritmo de busca binária conforme Seção 3 do PDF
    Retorna o índice i tal que xi[i] <= x_estrela < xi[i+1]
    """
    m = 0
    M = len(xi) - 1
    
    while M - m > 1:
        k = (M + m) // 2
        if x_estrela >= xi[k]:
            m = k
        else:
            M = k
    return m

def calcular_hi(xi):
    """
    Calcula os valores hi = xi+1 - xi
    """
    hi = []
    for j in range(len(xi) - 1):
        hi.append(xi[j+1] - xi[j])
    return hi

def calcular_tn_zn(hi, n):
    """
    Calcula os coeficientes tn (mi) e zn (lambda)
    conforme definidos no sistema de equações (2)
    """
    tn = []  # mi
    zn = []  # lambda
    
    # Calcula mi (tn)
    for j in range(n - 2):
        tn.append(hi[j] / (hi[j] + hi[j+1]))
    tn.append(0)
    
    # Calcula lambda (zn)
    zn.append(0)
    for j in range(n - 2):
        zn.append(hi[j+1] / (hi[j] + hi[j+1]))
    
    return tn, zn

def calcular_di(hi, yi, y0_pp, yn_pp):
    """
    Calcula o vetor de termos independentes d
    conforme sistema de equações (2)
    """
    n = len(yi)
    di = [2 * y0_pp]  # d0 = 2*y0''
    
    for j in range(n - 2):
        termo1 = (yi[j+2] - yi[j+1]) / hi[j+1]
        termo2 = (yi[j+1] - yi[j]) / hi[j]
        di.append((6 / (hi[j] + hi[j+1])) * (termo1 - termo2))
    
    di.append(2 * yn_pp)  # dn = 2*yn''
    return di

def construir_matriz_M(tn, zn, n):
    """
    Constrói a matriz tridiagonal M do sistema linear
    conforme equação (2)
    """
    M = [[0 for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 2
            elif i == j - 1:
                M[i][j] = zn[i]  # lambda
            elif i == j + 1:
                M[i][j] = tn[i-1]  # mi
    
    return M

def calcular_ai_bi(xi, yi, hi, M_sol):
    """
    Calcula as constantes Ai e Bi da fórmula do spline cúbico
    conforme equações (3)
    """
    n = len(xi)
    Ai = []
    Bi = []
    
    # Calcula Bi
    for p in range(n - 1):
        Bi.append(float(yi[p] - M_sol[p] * hi[p] * hi[p] / 6))
    
    # Calcula Ai
    for q in range(n - 1):
        termo1 = (yi[q+1] - yi[q]) / hi[q]
        termo2 = (M_sol[q+1] - M_sol[q]) * hi[q] / 6
        Ai.append(float(termo1 - termo2))
    
    return Ai, Bi

def avaliar_spline(xi, hi, M_sol, Ai, Bi, x_estrela):
    """
    Avalia o spline cúbico no ponto x_estrela
    usando a fórmula (1)
    """
    idx = busca_binaria(xi, x_estrela)
    
    termo1 = (M_sol[idx] / (6 * hi[idx])) * ((xi[idx+1] - x_estrela) ** 3)
    termo2 = (M_sol[idx+1] / (6 * hi[idx])) * ((x_estrela - xi[idx]) ** 3)
    termo3 = Ai[idx] * (x_estrela - xi[idx])
    
    return float(termo1 + termo2 + termo3 + Bi[idx])

def main():
    """
    Função principal para a Tarefa 1
    Implementa cálculo de splines cúbicos para dados da Tabela 1
    """
    # Dados da Tabela 1
    xi = [-0.9, -0.83, -0.6, -0.49, 0, 0.2, 0.6, 0.83]
    yi = [0.0, 1, 2.4, 4.1, 6, 8.2, 10.6, 13.4]
    y0_pp = 1   # y0''
    yn_pp = -1  # yn''
    
    n = len(xi)
    
    print("=== TAREFA 1 - CÁLCULO DE SPLINES CÚBICOS ===")
    print()
    
    # 1. Calcular hi
    hi = calcular_hi(xi)
    print("1. Valores de hi (diferenças entre xi consecutivos):")
    print("hi =", hi)
    print()
    
    # 2. Calcular tn (mi) e zn (lambda)
    tn, zn = calcular_tn_zn(hi, n)
    print("2. Coeficientes tn (mi) e zn (lambda):")
    print("tn =", tn)
    print("zn =", zn)
    print()
    
    # 3. Calcular vetor d (termos independentes)
    di = calcular_di(hi, yi, y0_pp, yn_pp)
    print("3. Vetor de termos independentes d:")
    print("di =", di)
    print()
    
    # 4. Construir matriz M do sistema linear
    M = construir_matriz_M(tn, zn, n)
    print("4. Matriz T do sistema linear:")
    for linha in M:
        print(linha)
    print()
    
    # 5. Resolver sistema com eliminação de Gauss
    M_sol = eliminacao_gauss([linha[:] for linha in M], di[:])
    print("5. Solução do sistema - coeficientes M:")
    print("M =", M_sol)
    print()
    
    # 6. Calcular constantes Ai e Bi
    Ai, Bi = calcular_ai_bi(xi, yi, hi, M_sol)
    print("6. Constantes Ai e Bi:")
    print("Ai =", Ai)
    print("Bi =", Bi)
    print()
    
    # 7. Avaliar spline em pontos de teste
    pontos_teste = [-0.6, 0.25, 0.5]
    print("7. Avaliação do spline em pontos de teste:")
    for x_estrela in pontos_teste:
        idx = busca_binaria(xi, x_estrela)
        S_val = avaliar_spline(xi, hi, M_sol, Ai, Bi, x_estrela)
        print(f"x* = {x_estrela}")
        print(f"Índice i = {idx}")
        print(f"S({x_estrela}) = {S_val}")
        print()

if __name__ == "__main__":
    main()