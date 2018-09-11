import math, copy

import numpy as np

from src.individual import Individual
from src import tree

class GeneticProgramming:
    def __init__(self, train, test, nb_individuals, nb_generations, p_crossover,
        max_tree_depth, tournament_size, random_generator):
        self.train           = train
        self.test            = test
        self.nb_individuals  = nb_individuals
        self.nb_generations  = nb_generations
        self.max_tree_depth  = max_tree_depth
        self.p_crossover     = p_crossover
        self.p_mutation      = 1 - p_crossover
        self.tournament_size = tournament_size
        self.rng             = random_generator

        self.X_train = train[:, :-1]
        self.y_train = train[:,-1]
        self.X_test  = test[:, :-1]
        self.y_test  = test[:, -1]
        self.n_features     = len(self.X_train[0])

        self.node_list = {
            'all': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply, tree.Value, tree.Variable ],
            'functions': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply ],
            'terminals': [ tree.Value, tree.Variable ]
        }

    def _init_population(self):
        population = []

        # Create 1/2 using Grow and 1/2 using Full
        for i in range(0, math.ceil(self.nb_individuals / 2)):
            i = Individual(self.max_tree_depth, self.n_features)
            i.grow(self.node_list, self.rng)
            population.append(i)
        
        for i in range(0, math.floor(self.nb_individuals / 2)):
            i = Individual(self.max_tree_depth, self.n_features)
            i.full(self.node_list, self.rng)
            population.append(i)

        return population

    def _tournament(self, population):
        # Generate a list of numbers from 0 to len(population)
        index_indivuals = np.arange(len(population))
        # Shuffle the list
        np.random.shuffle(index_indivuals)
        # Select the individuals that are selected
        index_indivuals = index_indivuals[:self.tournament_size]
        # Create a new list with this individuals
        selected = [ population[i] for i in index_indivuals]
        # Sort by fitness
        selected.sort(key=lambda x: x.fitness)
        # Select the best (lowest)
        return selected[0]

    def _selection(self):
        return
    
    def _crossover(self, ind1, ind2):
        return

    def _mutation(self, ind1):
        return

    def _fitness(self, population):
        # normalization = np.sum( np.power(self.y_train - np.mean(self.y_train), 2))
        normalization = len(self.y_train)

        for p in population:
            y_pred = p.predict(self.X_train)
            error_squared = np.sum(np.power(self.y_train - y_pred, 2))
            fitness_p = np.sqrt( error_squared / normalization )
            # TO-DO Check this one
            if (np.isinf(fitness_p)):
                fitness_p = 99999999
            p.fitness = fitness_p

    def run(self):
        scores = {
            'Train': [],
            'Test': []
        }

        # Generate initial population
        population = self._init_population()
        self._fitness(population)

        for i in range(0, self.nb_generations):
            new_population = []
            while ( len(new_population) < self.nb_individuals ):
                if (self.rng.random_sample() < self.p_crossover):
                    # Do Crossover
                    ind1 = self._tournament(population)
                    ind2 = self._tournament(population)
                    print('cross')
                    pass
                else:
                    # Do Mutation
                    print('muta')
                    pass
            # print(i)

        # Fitness

        # Returns fitness of every individual at generation i for train and test set
        return scores