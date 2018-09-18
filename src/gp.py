import math, copy, time

import numpy as np
import matplotlib.pyplot as plt

from src.individual import Individual
from src import tree

class GeneticProgramming:
    def __init__(self, train, test, nb_individuals, nb_generations,
        p_crossover, p_mutation, p_reproduction, max_tree_depth,
        tournament_size, elitist_operators, random_generator):
        self.train           = train
        self.test            = test
        self.nb_individuals  = nb_individuals
        self.nb_generations  = nb_generations
        self.max_tree_depth  = max_tree_depth
        self.p_crossover     = p_crossover
        self.p_mutation      = p_mutation
        self.p_reproduction  = p_reproduction
        self.tournament_size = tournament_size
        self.elitist_operators = elitist_operators
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
        individuals_per_level = self.nb_individuals / self.max_tree_depth
        
        # Create 1/2 using Grow and 1/2 using Full
        
        for max_depth in range(1, self.max_tree_depth+1):
            for j in range (0, int(individuals_per_level / 2)):
                indi_grow = Individual(self.max_tree_depth, self.n_features)
                indi_grow.grow(self.node_list, self.rng, 0, max_depth)
                population.append(indi_grow)

                indi_full = Individual(self.max_tree_depth, self.n_features)
                indi_full.full(self.node_list, self.rng, max_depth)
                population.append(indi_full)

        while ( len(population) < self.nb_individuals):
            indi_grow = Individual(self.max_tree_depth, self.n_features)
            indi_grow.grow(self.node_list, self.rng, 0, self.max_tree_depth)
            population.append(indi_grow)

        # print(len(population))
        # for i, p in enumerate(population):
        #     print(i, p.tree.getMaxDepth())
        # input()

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
        # print('crossover')
        # print('ind1', ind1.tree)
        # print('ind2', ind2.tree)

        son1 = copy.deepcopy(ind1)
        son2 = copy.deepcopy(ind2)
        son1.fitness = None
        son2.fitness = None

        random_node_1 = son1.select_random_node(self.rng)
        random_node_2 = son2.select_random_node(self.rng)
        while ( ((random_node_1['cur_depth'] + random_node_2['node_depth']) > self.max_tree_depth) or
                ((random_node_2['cur_depth'] + random_node_1['node_depth']) >= self.max_tree_depth)):
            random_node_1 = son1.select_random_node(self.rng)
            random_node_2 = son2.select_random_node(self.rng)

        
        
        # if (random_node_1['cur_depth'] + random_node_2['node_depth'] <= self.max_tree_depth):
        self._swap_nodes( copy.deepcopy(random_node_2['node']), random_node_1['node'] )

        # if (random_node_2['cur_depth'] + random_node_1['node_depth'] <= self.max_tree_depth):
        self._swap_nodes( copy.deepcopy(random_node_1['node']), random_node_2['node'] )


        # if (son1.tree.getMaxDepth() > 6):
        #     print(son1.tree.getMaxDepth())
        #     print(son1.tree)
        #     print(random_node_1)
        #     print(random_node_2)
        #     input()

        # print('ind1', ind1.tree)
        # print('ind2', ind2.tree)
        # print('son1', son1.tree)
        # print('son2', son2.tree)
        # input()
    
        return son1, son2

    def _mutation(self, individual):
        # print('indi1', individual.tree)
        new_individual = copy.deepcopy(individual)
        new_individual.fitness = None

        random_node = new_individual.select_random_node(self.rng)
        old_node_type = random_node['node'].type

        sub_tree = Individual(max_depth=self.max_tree_depth, n_features=self.n_features)
        sub_tree.grow(self.node_list, self.rng, random_node['cur_depth'], self.max_tree_depth)
        
        mutated_node = copy.deepcopy(sub_tree.tree.root)
        # print('mutaded_node', mutated_node)

        self._swap_nodes(mutated_node, random_node['node'])

        # print('indi1', individual.tree)
        # print('new', new_individual.tree)
        # input()

        # new_individual.node_list = None
        return new_individual

    def _fitness(self, population, X, y, substitute_fitness=True):
        normalization = np.sum( np.power(y - np.mean(y), 2))
        # normalization = np.std(self.y_train)
        fitness_list = []
        for p in population:
            y_pred = p.predict(X)
            # print(y_pred)
            error = np.sqrt(np.sum(np.power(y - y_pred, 2)) / normalization) #* ( (p.tree.getMaxDepth() / 6))
            print(error)
            # print(y_pred)
            # print(self.y_train)
            # print(y_pred-self.y_train)
            # print(np.power(y_pred - self.y_train, 2))
            # input()
            # error = np.sqrt(np.sum(np.power(y_pred - self.y_train, 2)) / len(y_pred))
            fitness_list.append(error)
            if (substitute_fitness):
                p.fitness = error

        return fitness_list
    def run(self):
        scores = {
            'Train': {
                'Best': [],
                'Average': [],
                'Worst': []
            },
            'Test': {
                'Best': [],
                'Average': [],
                'Worst': []
            }
        }

        # Generate initial population
        population = self._init_population()
        self._fitness(population, self.X_train, self.y_train)

        for gen_i in range(0, self.nb_generations):
            if (gen_i % 50 == 0):
                print('Generation', gen_i)
            new_population = []
            while ( len(new_population) < self.nb_individuals ):
                
                choice = self.rng.choice( ['crossover', 'mutation', 'reproduction'],
                    p=[self.p_crossover, self.p_mutation, self.p_reproduction])
                # print(choice)
                if (choice == 'crossover'):
                    # Do Crossover
                    ind1 = self._tournament(population)
                    ind2 = self._tournament(population)
                    son1, son2 = self._crossover(ind1, ind2)
                    self._fitness([son1, son2], self.X_train, self.y_train)

                    new_population.append(son1)
                    new_population.append(son2)
                elif (choice == 'mutation'):
                    # Do Mutation
                    individual = self._tournament(population)
                    mutated_individual = self._mutation(individual)
                    self._fitness([mutated_individual], self.X_train, self.y_train)
                    new_population.append(mutated_individual)
                else:
                    reproduct = self._tournament(population)
                    new_population.append(reproduct)
                    # print('hi')

            fitness_train = []
            best_individual = new_population[0]
            for p in new_population:
                if (p.fitness < best_individual.fitness):
                    best_individual = p
                fitness_train.append(p.fitness)
            fitness_train = np.array(fitness_train)
            fitness_train = fitness_train[fitness_train < np.percentile(fitness_train, 80)]

            if (gen_i % 50 == 0):
                sizes = []
                for p in new_population:
                    sizes.append(p.tree.getMaxDepth())
                print('Mean size of individuals', np.mean(sizes))
                print('Mean fitness', np.mean( fitness_train ))
                print('Best fitness', np.min( fitness_train ))

            scores['Train']['Average'].append(np.mean(fitness_train) )
            scores['Train']['Best'].append(np.min(fitness_train))
            scores['Train']['Best Individiual'] = best_individual.tree.__str__()
            
            if (gen_i == self.nb_generations-1):
                fitness_test = self._fitness(new_population, self.X_test, self.y_test, substitute_fitness=False)
                fitness_test = np.array(fitness_test)
                fitness_test = fitness_test[fitness_test < np.percentile(fitness_test, 80)]
                scores['Test']['Average'].append(np.mean(fitness_test) )
                scores['Test']['Best'].append(np.min(fitness_test))

            population = new_population

        # plt.plot(scores['Train'])
        # plt.show()

        # Fitness

        # Returns fitness of every individual at generation i for train and test set
        return scores