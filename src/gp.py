import math

from src.individual import Individual
from src import tree

class GeneticProgramming:
    def __init__(self, train, test, nb_individuals, nb_generations, p_crossover, max_tree_depth):
        self.train = train
        self.test  = test
        self.nb_individuals = nb_individuals
        self.nb_generations = nb_generations
        self.max_tree_depth = max_tree_depth
        self.p_crossover    = p_crossover
        self.p_mutation     = 1 - p_crossover
        self.n_features     = len(train[0])-1

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
            i.grow(self.node_list)
            population.append(i)
        
        for i in range(0, math.floor(self.nb_individuals / 2)):
            i = Individual(self.max_tree_depth, self.n_features)
            i.full(self.node_list)
            population.append(i)

        return population

    def _selection(self):
        return
    
    def _crossover(self, ind1, ind2):
        return
    
    ## TO-DO: Check this one out
    def _reproduction(self, ind1, ind2):
        return 

    def _mutation(self, ind1):
        return
    
    def run(self):
        scores = {
            'Train': [],
            'Test': []
        }

        population = self._init_population()

        for i in range(0, self.nb_generations):
            # print(i)
            pass

        # Fitness

        # Returns fitness of every individual at generation i for train and test set
        return scores