import random
import pandas as pd


class GeneticAlgorithm:
    """
    Genetic Algorithm class.
    """
    def __init__(self, data: dict, population_size: int, dna_size: int, mutation_probability: float):
        self.population_size = population_size
        self.data = data
        self.dna_size = dna_size
        self.mutation_probability = mutation_probability
        self.__create_population()

    def __create_population(self):
        self.population = [random.sample(self.data.keys(), self.dna_size) for _ in range(self.population_size)]

    def __fitness(self, dna):
        return 1/abs(sum([self.data[index] for index in dna]) - self.target)

    def __crossover(self, dna1, dna2):
        position = int(random.random() * self.dna_size)
        return dna1[:position] + dna2[position:], dna2[:position] + dna1[position:]

    def __mutate(self, dna):
        return [gene if random.random() > self.mutation_probability else random.choice(list(self.data.keys())) for gene in dna]

    def __dna_value(self, dna):
        return sum([self.data[index] for index in dna])

    def __assign_weights(self):
        weights = []
        total_fitness = sum([self.__fitness(dna) for dna in self.population])
        for dna in self.population:
            fitness = self.__fitness(dna)
            weights.append(fitness/total_fitness if total_fitness > 0 else 1)
        return weights

    def __find_best_solution(self):
        values = [self.__dna_value(dna) for dna in self.population]
        best_value = min(values, key=lambda x: abs(x - self.target))
        best_combination = self.population[values.index(best_value)]
        print(f'The best solution is : {best_value} with a combination of {best_combination}')
        return best_value

    def optimise(self, target, generations):
        # Setting the target
        self.target = target
        self.solutions = []
        # Optimisation start
        for _ in range(generations):
            # Print the best solution so far
            bestValue = self.__find_best_solution()
            self.solutions.append(bestValue)
            nest_generation = []

            # Assign weights based on fitness
            weights = self.__assign_weights()

            for _ in range(int(self.population_size / 2)):
                # Select parents
                new_parents = random.choices(self.population, weights, k = 2)

                # Create offsprings
                dna1, dna2 = self.__crossover(new_parents[0], new_parents[1])

                # Mutate
                dna1 = self.__mutate(dna1)
                dna2 = self.__mutate(dna2)

                # Add to new population
                nest_generation.append(dna1)
                nest_generation.append(dna2)

            # Replace population
            self.population = nest_generation


if __name__ == '__main__':
    print('Testing Genetic Algorithm Code: ')
    # Parameters
    populationSize = 50
    generations = 1000
    dnaSize = 10
    target = 4000
    mutationProbability = 0.01

    # Load data
    transactionsData = pd.read_csv('../data/dataTrans.csv').set_index('id').to_dict()['amount']

    # Genetic Algorithm
    ga = GeneticAlgorithm(transactionsData, populationSize, dnaSize, mutationProbability)

    # Generate starting population
    ga.optimise(target, generations)
