

from individual import Individual

class GeneticProgramming:
    def __init__(self, train, test, nb_generations, p_crossover, max_tree_depth):
        self.train = train
        self.test  = test
        self.nb_generations = nb_generations
        self.max_tree_depth = max_tree_depth
        self.p_crossover    = p_crossover
        self.p_mutation     = 1 - p_crossover

    def _init_population(self):
        return []

    def _selection(self):
        return
    
    def _crossover(self, ind1, ind2):
        return
    
    ## TO-DO: Check this one out
    def _reproduction(self, ind1, ind2)
        return 

    def _mutation(self, ind1):
        return
    
    def run(self):
        scores = {
            'Train': [],
            'Test': []
        }

        self.population = _init_population()

        for i in range(0, self.nb_generations):
            print(i)

        # Fitness

        # Returns fitness of every individual at generation i for train and test set
        return scores