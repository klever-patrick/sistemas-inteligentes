import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# Carregar os dados do CSV
data = pd.read_csv('CaixeiroSimples.csv', header=None)
points = data.values

# Função para calcular a distância euclidiana entre dois pontos
def euclidean_distance(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

# Função para calcular a distância total de uma rota
def total_distance(route):
    distance = 0
    for i in range(len(route) - 1):
        distance += euclidean_distance(points[route[i]], points[route[i + 1]])
    distance += euclidean_distance(points[route[-1]], points[route[0]])  # Voltar ao ponto de origem
    return distance

# Inicializar a população
def initialize_population(pop_size, num_points):
    population = []
    for _ in range(pop_size):
        individual = list(range(1, num_points))  # Ponto 0 é fixo, não incluímos aqui
        random.shuffle(individual)
        individual = [0] + individual  # Adiciona o ponto 0 no início
        population.append(individual)
    return population

# Seleção por torneio
def tournament_selection(population, fitnesses, k=3):
    selected = random.choices(list(zip(population, fitnesses)), k=k)
    selected = sorted(selected, key=lambda x: x[1])
    return selected[0][0]

# Operador de recombinação (crossover de dois pontos com variação)
def crossover(parent1, parent2):
    size = len(parent1)
    p1, p2 = [0]*size, [0]*size

    idx1, idx2 = sorted(random.sample(range(1, size), 2))
    p1_inher = parent1[idx1:idx2]
    p2_inher = [item for item in parent2 if item not in p1_inher]

    offspring1 = parent1[:idx1] + p2_inher[:len(p2_inher)//2] + p1_inher + p2_inher[len(p2_inher)//2:]
    offspring2 = parent2[:idx1] + p2_inher[len(p2_inher)//2:] + p1_inher + p2_inher[:len(p2_inher)//2]

    return offspring1, offspring2

# Operador de mutação
def mutate(individual, mutation_rate=0.01):
    for i in range(1, len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(1, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]

# Algoritmo Genético
def genetic_algorithm(pop_size, max_gens, elitism_size):
    num_points = len(points)
    population = initialize_population(pop_size, num_points)
    best_fitnesses = []

    for generation in range(max_gens):
        fitnesses = [total_distance(ind) for ind in population]
        new_population = []

        # Elitismo
        elites = sorted(list(zip(population, fitnesses)), key=lambda x: x[1])[:elitism_size]
        for elite in elites:
            new_population.append(elite[0])

        # Seleção, crossover e mutação
        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            offspring1, offspring2 = crossover(parent1, parent2)
            mutate(offspring1)
            mutate(offspring2)
            new_population.extend([offspring1, offspring2])

        population = new_population[:pop_size]
        best_fitness = min(fitnesses)
        best_fitnesses.append(best_fitness)
        print(f'Geração {generation+1}: Melhor distância = {best_fitness:.2f}')

    return best_fitnesses

# Parâmetros do algoritmo
POP_SIZE = 100
MAX_GENS = 500
ELITISM_SIZE = 5

# Executar o algoritmo genético
best_fitnesses = genetic_algorithm(POP_SIZE, MAX_GENS, ELITISM_SIZE)

# Exibir os resultados
plt.plot(best_fitnesses)
plt.xlabel('Geração')
plt.ylabel('Melhor Distância')
plt.title('Evolução da Melhor Distância')
plt.show()
