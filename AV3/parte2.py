import random
import time

class GeneticAlgorithm:
    def __init__(self, population_size=100):
        self.population_size = population_size
        self.population = self.create_population()
        self.found_solutions = set()

    def create_population(self):
        return [
            [random.randint(0, 7) for _ in range(8)]
            for _ in range(self.population_size)
        ]

    def calculate_fitness(self, individual):
        conflicts = 0
        for i in range(len(individual)):
            for j in range(i + 1, len(individual)):
                if individual[i] == individual[j] or abs(
                    individual[i] - individual[j]
                ) == abs(i - j):
                    conflicts += 1
        return 28 - conflicts

    def calculate_probabilities(self, fitnesses):
        total_fitness = sum(fitnesses)
        return [fitness / total_fitness for fitness in fitnesses]

    def crossover_single_point(self, parent1, parent2):
        cut_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:cut_point] + parent2[cut_point:]
        child2 = parent2[:cut_point] + parent1[cut_point:]
        return child1, child2

    def crossover_two_points(self, parent1, parent2):
        cut_point1 = random.randint(1, len(parent1) - 2)
        cut_point2 = random.randint(cut_point1 + 1, len(parent1) - 1)
        child1 = (
            parent1[:cut_point1] + parent2[cut_point1:cut_point2] + parent1[cut_point2:]
        )
        child2 = (
            parent2[:cut_point1] + parent1[cut_point1:cut_point2] + parent2[cut_point2:]
        )
        return child1, child2

    def crossover(self, parents):
        pc = random.uniform(0.85, 0.95)
        if random.random() <= pc:
            if random.choice([True, False]):
                return self.crossover_single_point(parents[0], parents[1])
            else:
                return self.crossover_two_points(parents[0], parents[1])
        else:
            return parents[0], parents[1]

    def mutation(self, individual):
        if random.random() <= 0.01:
            gene = random.randint(0, 7)
            new_value = random.randint(0, 7)
            individual[gene] = new_value

    def run(self):
        t = 0
        start_time = time.time()
        while len(self.found_solutions) < 92:
            fitnesses = [
                self.calculate_fitness(individual) for individual in self.population
            ]
            probabilities = self.calculate_probabilities(fitnesses)
            new_population = []
            for _ in range(self.population_size // 2):
                parents = []
                for _ in range(2):
                    r = random.random()
                    cumulative_prob = 0
                    i = 0
                    while cumulative_prob < r:
                        cumulative_prob += probabilities[i]
                        i += 1
                    parents.append(self.population[i - 1])

                child1, child2 = self.crossover(parents)
                self.mutation(child1)
                self.mutation(child2)
                new_population.extend([child1, child2])

            self.population = new_population
            t += 1

            for individual in self.population:
                if len(self.found_solutions) < 92:
                    solution = tuple(individual)
                    if solution not in self.found_solutions:
                        self.found_solutions.add(solution)
                        print(f"{solution}")

        total_time = time.time() - start_time
        return len(self.found_solutions), total_time

if __name__ == "__main__":
    ga = GeneticAlgorithm()
    num_solutions, elapsed_time = ga.run()
    print(f"Número de soluções diferentes encontradas: {num_solutions}")
    print(f"Tempo computacional: {elapsed_time} segundos")
