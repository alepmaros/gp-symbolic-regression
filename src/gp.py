import math, copy, time

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
        for i in range(0, math.ceil(self.nb_individuals * 0.8)):
            i = Individual(self.max_tree_depth, self.n_features)
            i.grow(self.node_list, self.rng)
            population.append(i)
        
        for i in range(0, math.floor(self.nb_individuals * 0.2)):
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

    def _swap_nodes(self, ind1, ind2):
        
        if ind2.parent.left == ind2:
            ind2.parent.left = ind1
        else:
            ind2.parent.right = ind1   
        ind1.parent = ind2.parent
    
    def _crossover(self, ind1, ind2):
        son1 = copy.deepcopy(ind1)
        son2 = copy.deepcopy(ind2)
        son1.fitness = None
        son2.fitness = None

        random_node_1 = son1.select_random_node(self.rng)
        random_node_2 = son2.select_random_node(self.rng)

        if (random_node_1['cur_depth'] + random_node_2['node_depth'] <= self.max_tree_depth):
            self._swap_nodes( copy.deepcopy(random_node_2['node']), random_node_1['node'] )

        if (random_node_2['cur_depth'] + random_node_1['node_depth'] <= self.max_tree_depth):
            self._swap_nodes( copy.deepcopy(random_node_1['node']), random_node_2['node'] )

        return son1, son2

    def _mutation(self, individual):
        new_individual = copy.deepcopy(individual)

        random_node = new_individual.select_random_node(self.rng)
        old_node_type = random_node['node'].type

        # If it is the last node of the tree, you cant turn into a function
        if (random_node['cur_depth'] == self.max_tree_depth):
            r = self.rng.randint(0, len(self.node_list['terminals']))
            new_node = self.node_list['terminals'][r](random_node['node'].parent, self.n_features, self.rng)
        else:
            r = self.rng.randint(0, len(self.node_list['all']))
            new_node = self.node_list['all'][r](random_node['node'].parent, self.n_features, self.rng)

        if (random_node['position'] == 'left'):
            random_node['node'].parent.left  = new_node
        else:
            random_node['node'].parent.right = new_node

        # Need to expand it (TO-DO NEED TO ADD DEPTH CHECK HERE, WILL JUST ADD TWO NEW VARIABLES FOR NOW)
        if (old_node_type == 'Terminal' and new_node.type == 'Function'):
            r = self.rng.randint(0, len(self.node_list['terminals']))
            new_node.left  = self.node_list['terminals'][r](new_node, self.n_features, self.rng)
            r = self.rng.randint(0, len(self.node_list['terminals']))
            new_node.right = self.node_list['terminals'][r](new_node.parent, self.n_features, self.rng)

        if (old_node_type == 'Function' and new_node.type == 'Function'):
            new_node.left = random_node['node'].left
            new_node.right = random_node['node'].right

        if (old_node_type == 'Function' and new_node.type == 'Terminal'):
            new_node.left  = None
            new_node.right = None

        return new_individual

    def _fitness(self, population):
        normalization = np.sum( np.power(self.y_train - np.mean(self.y_train), 2))
        for p in population:
            y_pred = p.predict(self.X_train)
            error = np.sqrt(np.sum(np.power(self.y_train - y_pred, 2)) / normalization)
            p.fitness = error

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
            
            if (i % 10 == 0):
                sizes = []
                for p in new_population:
                    sizes.append(p.tree.getMaxDepth())
                print('Mean size of individuals', np.mean(sizes))

            fitness = []
            for p in new_population:
                fitness.append(p.fitness)
            scores['Train'].append(np.mean(fitness))
            population = new_population

        # plt.plot(scores['Train'])
        # plt.show()

        # Fitness

        # Returns fitness of every individual at generation i for train and test set
        return scores