import math, copy, time

import numpy as np
import matplotlib.pyplot as plt

from src.individual import Individual
from src import tree

class GeneticProgramming:
    def __init__(self, train, test, nb_individuals, nb_generations,
        p_crossover, p_mutation, p_reproduction, max_tree_depth,
        tournament_size, elitist_operators, allow_sin, random_generator):
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
            'all': [ tree.Sum, tree.Subtraction, tree.Division, tree.Multiply, tree.Value, tree.Variable, tree.Variable, tree.Variable ],
            'functions': [ tree.Sum, tree.Subtraction, tree.Division, tree.Multiply ],
            'terminals': [ tree.Value, tree.Variable ]
        }

        if (allow_sin):
            self.node_list = {
                'all': [ tree.Sum, tree.Subtraction, tree.Division, tree.Multiply, tree.Value, tree.Variable, tree.Variable, tree.Sin ],
                'functions': [ tree.Sum, tree.Subtraction, tree.Division, tree.Multiply ],
                'terminals': [ tree.Value, tree.Variable, tree.Sin ]
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

        return population

    def _tournament(self, population):
        # Generate a list of numbers from 0 to len(population)
        index_individuals = np.arange(len(population))
        # Shuffle the list
        self.rng.shuffle(index_individuals)
        # Select the individuals that are selected
        index_individuals = index_individuals[:self.tournament_size]
        # Create a new list with this individuals
        selected_individuals = [ population[i] for i in index_individuals]
        # Sort by fitness
        selected_individuals.sort(key=lambda x: x.fitness)
        # Select the best (lowest)
        return selected_individuals[0]

    def _select_k_best_individuals(self, population, k=5):
        # Create a new list with this individuals
        selected_individuals = population
        # Sort by fitness
        selected_individuals.sort(key=lambda x: x.fitness)
        # Select the best (lowest)
        return selected_individuals[0:k]

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

        while ( ((random_node_1['cur_depth'] + random_node_2['node_depth']) > self.max_tree_depth) or
                ((random_node_2['cur_depth'] + random_node_1['node_depth']) > self.max_tree_depth)):
            random_node_1 = son1.select_random_node(self.rng)
            random_node_2 = son2.select_random_node(self.rng)
        
        self._swap_nodes( copy.deepcopy(random_node_2['node']), random_node_1['node'] )
        self._swap_nodes( copy.deepcopy(random_node_1['node']), random_node_2['node'] )

        return son1, son2

    def _mutation(self, individual):
        new_individual = copy.deepcopy(individual)
        new_individual.fitness = None

        random_node = new_individual.select_random_node(self.rng)

        sub_tree = Individual(max_depth=self.max_tree_depth, n_features=self.n_features)
        sub_tree.grow(self.node_list, self.rng, random_node['cur_depth'], self.max_tree_depth)
        
        mutated_node = copy.deepcopy(sub_tree.tree.root)

        self._swap_nodes(mutated_node, random_node['node'])

        return new_individual

    def _fitness(self, population, X, y, substitute_fitness=True):
        normalization = np.sum( np.power(y - np.mean(y), 2))

        fitness_list = []
        for p in population:
            y_pred = p.predict(X)
            error = np.sqrt(np.sum(np.power(y - y_pred, 2)) / normalization)
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

                    if (not self.elitist_operators):
                        new_population.append(son1)
                        new_population.append(son2)
                    else:
                        add_better_parent = True
                        if (son1.fitness <= ind1.fitness and son1.fitness <= ind2.fitness):
                            new_population.append(son1)
                            add_better_parent = False
                        
                        if (son2.fitness <= ind1.fitness and son2.fitness <= ind2.fitness):
                            new_population.append(son2)
                            add_better_parent = False

                        if (add_better_parent):
                            if (ind1.fitness <= ind2.fitness):
                                copy_indi = copy.deepcopy(ind1)
                                self._fitness([copy_indi], self.X_train, self.y_train)
                                new_population.append(copy_indi)
                            else:
                                copy_indi = copy.deepcopy(ind2)
                                self._fitness([copy_indi], self.X_train, self.y_train)
                                new_population.append(copy_indi)

                elif (choice == 'mutation'):
                    # Do Mutation
                    individual = self._tournament(population)
                    mutated_individual = self._mutation(individual)
                    self._fitness([mutated_individual], self.X_train, self.y_train)
                    if (not self.elitist_operators):
                        new_population.append(mutated_individual)
                    else:
                        if (mutated_individual.fitness <= individual.fitness):
                            new_population.append(mutated_individual)
                        else:
                            copy_indi = copy.deepcopy(individual)
                            self._fitness([copy_indi], self.X_train, self.y_train)
                            new_population.append(copy_indi)
                else:
                    reproduct = self._tournament(population)
                    new_population.append(reproduct)
                    # print('hi')

            # Saving fitness for new population
            fitness_train = []
            best_individual = new_population[0]
            for p in new_population:
                if (p.fitness < best_individual.fitness):
                    best_individual = p
                fitness_train.append(p.fitness)
            fitness_train = np.array(fitness_train)
            fitness_train = fitness_train[fitness_train < 5.0]

            # if (gen_i % 50 == 0):
            #     sizes = []
            #     for p in new_population:
            #         sizes.append(p.tree.getMaxDepth())
            #     print('Mean size of individuals', np.mean(sizes))
            #     print('Mean fitness', np.mean( fitness_train ))
            #     print('Best fitness', np.min( fitness_train ))

            scores['Train']['Average'].append(np.mean(fitness_train) )
            scores['Train']['Best'].append(np.min(fitness_train))
            scores['Train']['Best Individual'] = best_individual.tree.__str__()
            
            # Calculate the fitness on the train set
            if (gen_i == self.nb_generations-1):
                # k_individuals = self._select_k_best_individuals(new_population, k=50)
                fitness_test = self._fitness(new_population, self.X_test, self.y_test, substitute_fitness=False)
                fitness_test = np.array(fitness_test)

                for index, fitt in enumerate(fitness_test):
                    if fitt == np.min(fitness_test):
                        scores['Test']['Best Individual'] = new_population[index].tree.__str__()

                scores['Test']['Average'].append(np.mean(fitness_test) )
                scores['Test']['Best'].append(np.min(fitness_test))

                score_best_individual_train = self._fitness([best_individual], self.X_test, self.y_test, substitute_fitness=False)
                scores['Test']['Best Individual From Train'] = score_best_individual_train[0]
            population = new_population

        # Returns fitness of every individual at generation i for train and test set
        return scores
