import numpy as np
import matplotlib.pyplot as plt
from typing import Union, Literal

problem = 1 # Variável para definir qual o problema a ser analisado
algorithm: Union[str, Literal['HillClimbing', 'LRS', 'GRS', 'SimulatedAnnealing']] = 'HillClimbing'  # Variável para setar o algoritmo
isToPlotTheGraph = True

# Definição de variáveis
MaximoIteracao = 1000
MaximoCandidatos = 10
Epsilon = 0.1
Sigma = 0.1
Alpha = 0.9
Temperatura = 10

# Definindo os limites com base no problema
if problem == 1:
    Low = -100
    High = 100
elif problem == 2:
    Low = -2
    High = 4
elif problem == 3:
    Low = -8
    High = 8
elif problem == 4:
    Low = -5.12
    High = 5.12
elif problem == 5:
    Low = -10
    High = 10
elif problem == 6:
    Low = -1
    High = 3
elif problem == 7:
    Low = 0
    High = np.pi
elif problem == 8:
    Low = -200
    High = 20

# Definindo se o problema é de minimização ou maximização
def is_maximization(problem):
    maximization_problems = [2, 5, 6]
    return problem in maximization_problems

def fn(x1, x2):
    if problem == 1:
        return x1**2 + x2**2
    elif problem == 2:
        return np.exp(-x1**2 - x2**2) + 2 * np.exp(-((x1 - 1.7)**2 + (x2 - 1.7)**2))
    elif problem == 3:
        return -20 * np.exp(-0.2 * np.sqrt(0.5 * (x1**2 + x2**2))) - np.exp(0.5 * (np.cos(2 * np.pi * x1) + np.cos(2 * np.pi * x2))) + 20 + np.exp(1)
    elif problem == 4:
        return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)
    elif problem == 5:
        return ((x1 * np.cos(x1)) / 20) + (2 * np.exp(-(x1**2) - (x2 - 1)**2)) + 0.01 * x1 * x2
    elif problem == 6:
        return x1 * np.sin(4 * np.pi * x1) - x2 * np.sin(4 * np.pi * x2 + np.pi) + 1
    elif problem == 7:
        return -np.sin(x1) * np.sin((x1**2) / np.pi)**2.10 - np.sin(x2) * np.sin((2 * x2**2) / np.pi)**2.10
    else:
        return -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47)))) - x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))


# Função para plotar o gráfico
def plotGraph(ax, hill_trajectory, lrs_trajectory, grs_trajectory):
    x1 = np.linspace(Low, High, MaximoIteracao)
    X1, X2 = np.meshgrid(x1, x1)
    Y = fn(X1, X2)
    
    ax.plot_surface(X1, X2, Y, rstride=10, cstride=10, alpha=0.6, cmap='viridis')

    hill_x1, hill_x2 = zip(*hill_trajectory)
    hill_z = [fn(x1, x2) + 0.1 for x1, x2 in hill_trajectory]  # Elevar os marcadores
    ax.plot(hill_x1, hill_x2, hill_z, marker='^', color='#1f77b4', label='Hill Climbing', alpha=0.8)

    lrs_x1, lrs_x2 = zip(*lrs_trajectory)
    lrs_z = [fn(x1, x2) + 0.1 for x1, x2 in lrs_trajectory]  # Elevar os marcadores
    ax.plot(lrs_x1, lrs_x2, lrs_z, marker='s', color='#2ca02c', label='Local Random Search', alpha=0.8)

    grs_x1, grs_x2 = zip(*grs_trajectory)
    grs_z = [fn(x1, x2) + 0.1 for x1, x2 in grs_trajectory]  # Elevar os marcadores
    ax.plot(grs_x1, grs_x2, grs_z, marker='o', color='#d62728', label='Global Random Search', alpha=0.8)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('Comparação das Trajetórias dos Algoritmos')
    ax.legend()

# Algoritmo Hill Climbing
def HillClimbing(low, high, epsilon, maxIt, maxCad, maximization):
    x1Best = np.random.uniform(low, high)
    x2Best = np.random.uniform(low, high)
    fBest = fn(x1Best, x2Best)
    trajectory = [(x1Best, x2Best)]
    
    i = 0
    while i < maxIt:
        improvement = False
        
        for j in range(maxCad):
            candidate1 = x1Best + np.random.uniform(-epsilon, epsilon)  # Gera um candidato vizinho
            candidate2 = x2Best + np.random.uniform(-epsilon, epsilon)  # Gera um candidato vizinho
            
            # Verificação de limites (restrição em caixa)
            if candidate1 < low or candidate1 > high or candidate2 < low or candidate2 > high:
                continue
            
            F = fn(candidate1, candidate2)
            
            if (maximization and F > fBest) or (not maximization and F < fBest):  # Procurando maximizar ou minimizar a função
                x1Best, x2Best = candidate1, candidate2
                fBest = F
                improvement = True
                trajectory.append((x1Best, x2Best))
        
        if not improvement:
            epsilon *= 0.9  # Reduz o tamanho do passo se não houver melhoria
        i += 1

    return [x1Best, x2Best], fBest, trajectory

# Algoritmo Local Random Search (LRS)
def LocalRandomSearch(low, high, maxIt, sigma, maximization):
    xBest = np.random.uniform(low, high, size=2)  # Ponto inicial aleatório dentro do intervalo
    fBest = fn(*xBest)
    trajectory = [tuple(xBest)]
    
    i = 0
    while i < maxIt:
        n = np.random.normal(0, sigma, size=2)  # Perturbação
        
        xCand = xBest + n  # Candidato gerado
        
        # Verificação de limites (restrição em caixa)
        xCand = np.clip(xCand, low, high)
        
        fCand = fn(*xCand)
        
        if (maximization and fCand > fBest) or (not maximization and fCand < fBest):  # Procurando maximizar ou minimizar a função
            xBest = xCand
            fBest = fCand
            trajectory.append(tuple(xBest))
        
        i += 1
    
    return xBest, fBest, trajectory

# Algoritmo Global Random Search (GRS)
def GlobalRandomSearch(low, high, maxIt, maximization):
    xBest = np.random.uniform(low, high, size=2)  # Ponto inicial aleatório dentro do intervalo
    fBest = fn(*xBest)
    trajectory = [tuple(xBest)]
    
    i = 0
    while i < maxIt:
        xCand = np.random.uniform(low, high, size=2)  # Candidato gerado
        
        fcand = fn(*xCand)
        
        if (maximization and fcand > fBest) or (not maximization and fcand < fBest):  # Procurando maximizar ou minimizar a função
            xBest = xCand
            fBest = fcand
            trajectory.append(tuple(xBest))
        
        i += 1
    
    return xBest, fBest, trajectory

# Execução dos algoritmos
def RunAlgorithm(algorithm, maximization):
    if algorithm == 'HillClimbing':
        return HillClimbing(low=Low, high=High, epsilon=Epsilon, maxIt=MaximoIteracao, maxCad=10, maximization=maximization)
    elif algorithm == 'LRS':
        return LocalRandomSearch(low=Low, high=High, maxIt=MaximoIteracao, sigma=Sigma, maximization=maximization)
    elif algorithm == 'GRS':
        return GlobalRandomSearch(low=Low, high=High, maxIt=MaximoIteracao, maximization=maximization)
    else:
        raise ValueError("Algoritmo não reconhecido")

# Determinar se o problema é de maximização ou minimização
maximization = is_maximization(problem)

# Coleta dos resultados e trajetórias
hill_result, hill_fbest, hill_trajectory = RunAlgorithm('HillClimbing', maximization)
lrs_result, lrs_fbest, lrs_trajectory = RunAlgorithm('LRS', maximization)
grs_result, grs_fbest, grs_trajectory = RunAlgorithm('GRS', maximization)

print(f"Hill Climbing: x1, x2 = {hill_result}, valor da função = {hill_fbest}")
print(f"LRS: x1, x2 = {lrs_result}, valor da função = {lrs_fbest}")
print(f"GRS: x1, x2 = {grs_result}, valor da função = {grs_fbest}")

# Plotar os resultados em um único gráfico
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

plotGraph(ax, hill_trajectory, lrs_trajectory, grs_trajectory)

plt.show()
