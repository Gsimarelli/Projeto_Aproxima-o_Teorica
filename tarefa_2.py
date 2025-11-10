import numpy as np
import math

# Parâmetros do problema
A = 0.0  # Início do intervalo [a, b]
B = 1.0  # Fim do intervalo [a, b]
N = 10   # Número de subintervalos (n)
M = 13   # Número de subintervalos para os pontos de teste (m)
H = (B - A) / N  # Passo de interpolação (h)

# Função a ser interpolada: f(x) = exp(x)
def f_exp(x):
    return math.exp(x)

# Função de teste: f(x) = cos(x)
def f_cos(x):
    return math.cos(x)

def calcular_nos_e_valores(f, n, a, h):
    """Calcula os nós (x_i) e os valores da função (y_i) nos nós."""
    x = np.array([a + i * h for i in range(n + 1)])
    y = np.array([f(xi) for xi in x])
    return x, y

def solucao_sistema_tridiagonal_gauss(a, b, c, d):
    """
    Resolve um sistema linear tridiagonal A*x = d, usando a lógica da
    Eliminação Gaussiana com Substituição Retroativa, otimizada para
    matrizes tridiagonais.

    a: subdiagonal (a_i)
    b: diagonal principal (b_i)
    c: superdiagonal (c_i)
    d: vetor do lado direito (d_i)
    """
    n = len(d)
    
    # Vetores temporários para armazenar os coeficientes modificados
    # O vetor b será modificado para armazenar a nova diagonal principal (p)
    # O vetor d será modificado para armazenar o novo lado direito (q)
    
    # 1. Eliminação Progressiva (Forward Elimination)
    # Transforma a matriz em uma matriz triangular superior
    
    # O loop começa em i=1 porque a primeira linha (i=0) não precisa de eliminação
    for i in range(1, n):
        # Fator de eliminação (m = a[i-1] / b[i-1])
        m = a[i-1] / b[i-1]
        
        # Atualiza o elemento da diagonal principal (b_i)
        # b[i] = b[i] - m * c[i-1]
        b[i] = b[i] - m * c[i-1]
        
        # Atualiza o elemento do lado direito (d_i)
        # d[i] = d[i] - m * d[i-1]
        d[i] = d[i] - m * d[i-1]
        
    # 2. Substituição Retroativa (Backward Substitution)
    # Calcula as incógnitas de trás para frente
    x = np.zeros(n)
    
    # Última variável (x_{n-1})
    x[n-1] = d[n-1] / b[n-1]
    
    # Para i = n-2 até 0
    for i in range(n - 2, -1, -1):
        # x_i = (d_i - c_i * x_{i+1}) / b_i
        x[i] = (d[i] - c[i] * x[i+1]) / b[i]
        
    return x

def calcular_coeficientes_spline_projeto(x, y, y0_linha_linha, yn_linha_linha):
    """
    CORREÇÃO: Calcula os coeficientes do spline conforme a formulação EXATA do projeto
    usando as equações (2) e (3)
    """
    n = len(x) - 1
    
    # 1. Calcular h_i
    h = [x[i+1] - x[i] for i in range(n)]
    
    # 2. Calcular μ_i, λ_i, d_i conforme equação (2) do projeto
    mu = [0.0] * (n+1)      # μ₀, μ₁, ..., μₙ
    lam = [0.0] * (n+1)     # λ₀, λ₁, ..., λₙ  
    d = [0.0] * (n+1)       # d₀, d₁, ..., dₙ
    
    # Condições de contorno (linha 1 do sistema)
    mu[0] = 0.0
    lam[0] = 0.0
    d[0] = 2 * y0_linha_linha
    
    # Para i = 1 até n-1 (linhas internas do sistema)
    for i in range(1, n):
        denominador = h[i-1] + h[i]
        mu[i] = h[i-1] / denominador
        lam[i] = h[i] / denominador
        d[i] = (6.0 / denominador) * (
            ((y[i+1] - y[i]) / h[i]) - 
            ((y[i] - y[i-1]) / h[i-1])
        )
    
    # Condições de contorno (última linha do sistema)
    mu[n] = 0.0
    lam[n] = 0.0
    d[n] = 2 * yn_linha_linha
    
    # 3. Montar matriz tridiagonal do sistema: [μ, 2, λ]
    # A diagonal principal é sempre 2 (conforme equação 2 do projeto)
    diagonal_principal = [2.0] * (n+1)
    
    # 4. Resolver sistema tridiagonal [μ, 2, λ] * M = d
    # Usando a função de eliminação de Gauss que já temos
    M = solucao_sistema_tridiagonal_gauss(
        mu[1:],           # subdiagonal (μ₁ até μₙ)
        diagonal_principal.copy(),  # diagonal principal (2, 2, ..., 2)
        lam[:-1],         # superdiagonal (λ₀ até λₙ₋₁)  
        d.copy()          # vetor independente
    )
    
    # 5. Calcular A_i e B_i conforme equação (3) do projeto
    A_coef = []
    B_coef = []
    
    for i in range(n):
        B_i = y[i] - (M[i] / 6.0) * (h[i]**2)
        A_i = ((y[i+1] - y[i]) / h[i]) - ((M[i+1] - M[i]) / 6.0) * h[i]
        A_coef.append(A_i)
        B_coef.append(B_i)
    
    return M, A_coef, B_coef, h

def busca_binaria(x_nos, x_ponto):
    """
    Implementa o algoritmo de busca binária da Seção 3.2 do projeto
    """
    n = len(x_nos) - 1
    m = 0
    M = n
    
    if x_ponto < x_nos[0] or x_ponto > x_nos[n]:
        return -1
    
    while (M - m) > 1:
        k = (M + m) // 2
        
        if x_ponto > x_nos[k]:
            m = k
        else:
            M = k
    
    return m

def avaliar_spline_corrigido(x_ponto, x_nos, y_nos, M, A, B, h):
    """
    Avalia o Spline Cúbico usando a fórmula EXATA do projeto
    e busca binária para eficiência
    """
    n = len(x_nos) - 1
    
    i = busca_binaria(x_nos, x_ponto)
    
    if i == -1:
        return np.nan
    
    xi = x_nos[i]
    xi1 = x_nos[i+1]
    hi = h[i]
    
    termo1 = (M[i] / (6.0 * hi)) * (xi1 - x_ponto)**3
    termo2 = (M[i+1] / (6.0 * hi)) * (x_ponto - xi)**3
    termo3 = A[i] * (x_ponto - xi)
    termo4 = B[i]
    
    S_x = termo1 + termo2 + termo3 + termo4
           
    return S_x

def executar_analise_corrigida(f, a, b, n, m, nome_funcao, y0_linha_linha=0, yn_linha_linha=0):
    """Função principal para executar a análise e calcular o erro."""
    
    h_total = (b - a) / n
    h_star = (b - a) / m
    
    print(f"\n--- Análise para f(x) = {nome_funcao} ---")
    print(f"Parâmetros: n={n}, m={m}, h={h_total:.4f}, h*={h_star:.4f}")
    print(f"Condições contorno: y0''={y0_linha_linha}, yn''={yn_linha_linha}")
    
    x_nos, y_nos = calcular_nos_e_valores(f, n, a, h_total)
    
    M, A_coef, B_coef, h_intervalos = calcular_coeficientes_spline_projeto(
        x_nos, y_nos, y0_linha_linha, yn_linha_linha
    )
    print(f"M_0 a M_n (M_i): {M}")
    
    t_star = np.array([a + j * h_star for j in range(m + 1)])
    
    S_t = np.array([
        avaliar_spline_corrigido(tj, x_nos, y_nos, M, A_coef, B_coef, h_intervalos) 
        for tj in t_star
    ])
    f_t = np.array([f(tj) for tj in t_star])
    
    e_j = np.abs(f_t - S_t)
    E_n = np.max(e_j)
    
    print("\nResultados nos Pontos de Teste (j=0 a m):")
    print(f"{'j':<5} | {'t_j*':<15} | {'f(t_j*)':<20} | {'S_Delta(t_j*)':<20} | {'e_j':<20}")
    print("-" * 85)
    
    for j in range(m + 1):
        print(f"{j:<5} | {t_star[j]:<15.10f} | {f_t[j]:<20.15f} | {S_t[j]:<20.15f} | {e_j[j]:<20.15e}")

    print("-" * 85)
    print(f"Erro Máximo (E_n): {E_n:.15e}")
    
    return t_star, S_t, E_n

def main():
    """Função principal da Tarefa 2"""
    print("=" * 80)
    print("RESOLUÇÃO DA TAREFA 2: f(x) = exp(x)")
    print("=" * 80)
    
    # Executar análise para f(x) = exp(x)
    t_exp, S_exp, E_exp = executar_analise_corrigida(
        f_exp, A, B, N, M, "exp(x)", 
        y0_linha_linha=0,
        yn_linha_linha=0
    )
    
    print("\n" + "=" * 80)
    print("TESTE DE VALIDAÇÃO: f(x) = cos(x)")
    print("=" * 80)
    
    # Executar análise de validação para f(x) = cos(x)
    t_cos, S_cos, E_cos = executar_analise_corrigida(
        f_cos, A, B, N, M, "cos(x)", 
        y0_linha_linha=0,
        yn_linha_linha=0
    )

if __name__ == "__main__":
    main()