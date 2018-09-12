import math, copy

import numpy as np
import matplotlib.pyplot as plt

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
        index_individuals = np.arange(len(population))
        # Shuffle the list
        np.random.shuffle(index_individuals)
        # Select the individuals that are selected
        index_individuals = index_individuals[:self.tournament_size]
        # Create a new list with this individuals
        selected_individuals = [ population[i] for i in index_individuals]
        # Sort by fitness
        selected_individuals.sort(key=lambda x: x.fitness)
        # Select the best (lowest)
        return selected_individuals[0]
    
    def _crossover(self, ind1, ind2):
        son1 = copy.deepcopy(ind1)
        son2 = copy.deepcopy(ind2)

        random_node_1 = son1.select_random_node(self.rng)
        random_node_2 = son2.select_random_node(self.rng)

        parent_aux = random_node_1[1].parent
        random_node_1[1].parent = random_node_2[1].parent
        random_node_2[1].parent = parent_aux

        if (random_node_1[0] == 'left'):
            random_node_2[1].parent.left = random_node_2[1]
        else:
            random_node_2[1].parent.right = random_node_2[1]

        if (random_node_2[0] == 'left'):
            random_node_1[1].parent.left = random_node_1[1]
        else:
            random_node_1[1].parent.right = random_node_1[1]

        return son1, son2

    def _mutation(self, individual):
        new_individual = copy.deepcopy(individual)

        random_node = new_individual.select_random_node(self.rng)
        old_node_type = random_node[1].type

        r = self.rng.randint(0, len(self.node_list['all']))
        new_node = self.node_list['all'][r](random_node[1].parent, self.n_features, self.rng)
        if (random_node[0] == 'left'):
            random_node[1].parent.left  = new_node
        else:
            random_node[1].parent.right = new_node

        # Need to expand it (TO-DO NEED TO ADD DEPTH CHECK HERE, WILL JUST ADD TWO NEW VARIABLES FOR NOW)
        if (old_node_type == 'Terminal' and new_node.type == 'Function'):
            r = self.rng.randint(0, len(self.node_list['terminals']))
            new_node.left  = self.node_list['terminals'][r](new_node, self.n_features, self.rng)
            r = self.rng.randint(0, len(self.node_list['terminals']))
            new_node.right = self.node_list['terminals'][r](new_node.parent, self.n_features, self.rng)

        if (old_node_type == 'Function' and new_node.type == 'Function'):
            new_node.left = random_node[1].left
            new_node.right = random_node[1].right

        if (old_node_type == 'Function' and new_node.type == 'Terminal'):
            new_node.left  = None
            new_node.right = None

        return new_individual

    def _fitness(self, population):
        # TO DO FIX THIS FOR NORMALIZED VERSION
        normalization = np.sum( np.power(self.y_train - np.mean(self.y_train), 2))
        # normalization = len(self.y_train)

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
            if (i % 10 == 0):
                print('Generation', i)
            new_population = []
            while ( len(new_population) < self.nb_individuals ):
                if (self.rng.random_sample() < self.p_crossover):
                    # Do Crossover
                    ind1 = self._tournament(population)
                    ind2 = self._tournament(population)
                    son1, son2 = self._crossover(ind1, ind2)
                    self._fitness([son1, son2])
                    new_population.append(son1)
                    new_population.append(son2)
                else:
                    # Do Mutation
                    individual = self._tournament(population)
                    mutated_individual = self._mutation(individual)
                    self._fitness([mutated_individual])
                    new_population.append(mutated_individual)
            
            fitness = []
            for p in new_population:
                fitness.append(p.fitness)
            scores['Train'].append(np.mean(fitness))
            population = new_population

        plt.plot(scores['Train'])
        plt.show()

        # Fitness

        # Returns fitness of every individual at generation i for train and test set
        return scores