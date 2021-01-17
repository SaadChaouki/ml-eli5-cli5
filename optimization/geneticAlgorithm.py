import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class GeneticAlgorithm:
    """
    Genetic Algorithm class.
    """
    def __init__(self, data: dict, populationSize: int, dnaSize: int, mutationProbability: float):
        self.populationSize = populationSize
        self.data = data
        self.dnaSize = dnaSize
        self.mutationProbability = mutationProbability
        self.__createPopulation()

    def __createPopulation(self):
        self.population = [random.sample(self.data.keys(), self.dnaSize) for _ in range(self.populationSize)]

    def __fitness(self, dna):
        return 1/abs(sum([self.data[index] for index in dna]) - self.target)

    def __crossover(self, dna1, dna2):
        position = int(random.random() * self.dnaSize)
        return dna1[:position] + dna2[position:], dna2[:position] + dna1[position:]

    def __mutate(self, dna):
        return [gene if random.random() > self.mutationProbability else random.choice(list(self.data.keys())) for gene in dna]

    def __dnaValue(self, dna):
        return sum([self.data[index] for index in dna])

    def __assignWeights(self):
        weights = []
        totalFitness = sum([self.__fitness(dna) for dna in self.population])
        for dna in self.population:
            fitness = self.__fitness(dna)
            weights.append(fitness/totalFitness if totalFitness > 0 else 1)
        return weights

    def __findBestSolution(self):
        values = [self.__dnaValue(dna) for dna in self.population]
        bestValue = min(values, key=lambda x: abs(x - self.target))
        bestCombination = self.population[values.index(bestValue)]
        print(f'The best solution is : {bestValue} with a combination of {bestCombination}')
        return bestValue

    def optimise(self, target, generations):
        # Setting the target
        self.target = target
        self.solutions = []
        # Optimisation start
        for _ in range(generations):
            # Print the best solution so far
            bestValue = self.__findBestSolution()
            self.solutions.append(bestValue)
            nextGeneration = []

            # Assign weights based on fitness
            weights = self.__assignWeights()

            for _ in range(int(self.populationSize/2)):
                # Select parents
                newParents = random.choices(self.population, weights, k = 2)

                # Create offsprings
                dna1, dna2 = self.__crossover(newParents[0], newParents[1])

                # Mutate
                dna1 = self.__mutate(dna1)
                dna2 = self.__mutate(dna2)

                # Add to new population
                nextGeneration.append(dna1)
                nextGeneration.append(dna2)

            # Replace population
            self.population = nextGeneration


    def plotEvolution(self):
        plt.clf()
        sns.lineplot(x = range(len(self.solutions)), y = self.solutions)
        plt.show()


if __name__ == '__main__':
    print('Testing Genetic Algorithm Code: ')
    # Parameters
    populationSize = 50
    generations = 1000
    dnaSize = 10
    target = 500
    mutationProbability = 0.01

    # Load data
    transactionsData = pd.read_csv('../data/dataTrans.csv').set_index('id').to_dict()['amount']

    # Genetic Algorithm
    ga = GeneticAlgorithm(transactionsData, populationSize, dnaSize, mutationProbability)

    # Generate starting population
    ga.optimise(target, generations)

    # Plot the solutions
    ga.plotEvolution()
