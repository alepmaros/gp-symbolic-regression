import itertools

import numpy as np
from src import tree

class Individual:
    def __init__(self, max_depth=7, n_features=2):
        self.tree       = tree.Tree()
        self.max_depth  = max_depth
        self.n_features = n_features
        self.fitness    = None

    def predict(self, X):
        y_pred = self.tree.evalTree(X)
        return y_pred

    def select_random_node(self, rng):
        nodes = self.tree.getListOfNodes()
        return nodes[rng.randint(0, len(nodes))]


    #########################################
    # Methods of Creating New Individuals   #
    #########################################

    def _grow(self, parent, node, depth, node_list, rng):
        # If reached max depth, fill it with terminals
        if (depth >= self.max_depth):
            r = rng.randint(0, len(node_list['terminals']))
            node = node_list['terminals'][r](parent, self.n_features, rng)
            return node

        # If still hasnt reached max depth, keep expanding it
        r = rng.randint(0, len(node_list['all']))
        node = node_list['all'][r](parent, self.n_features, rng)
        if (node.type == 'Function'):
            node.left  = self._grow(node, node.left, depth+1, node_list, rng)
            node.right = self._grow(node, node.right, depth+1, node_list, rng)
        return node

    def grow(self, node_list, rng):
        # Choose a random Node to be added to the tree. The Root needs to be a function
        r = rng.randint(0, len(node_list['functions']))

        # Set the root of the tree to be that node
        self.tree.root = node_list['functions'][r](None, self.n_features, rng)

        self.tree.root.left  = self._grow(self.tree.root, self.tree.root.left, 1, node_list, rng)
        self.tree.root.right = self._grow(self.tree.root, self.tree.root.right, 1, node_list, rng)

    def _full(self, parent, node, depth, node_list, rng):
        # If reached max depth, fill it with terminals
        if (depth >= self.max_depth):
            r = rng.randint(0, len(node_list['terminals']))
            node = node_list['terminals'][r](parent, self.n_features, rng)
            return node

        # If still hasnt reached max depth, keep expanding with functions
        r = rng.randint(0, len(node_list['functions']))
        node = node_list['functions'][r](parent, self.n_features, rng)
        node.left  = self._full(node, node.left, depth+1, node_list, rng)
        node.right = self._full(node, node.right, depth+1, node_list, rng)
        return node

    def full(self, node_list, rng):
        # Choose a random Function to be added to the tree
        r = rng.randint(0, len(node_list['functions']))
        self.tree.root = node_list['functions'][r](None, self.n_features, rng)
        self.tree.root.left  = self._full(self.tree.root, self.tree.root.left, 1, node_list, rng)
        self.tree.root.right = self._full(self.tree.root, self.tree.root.right, 1, node_list, rng)


# node_list = {
#     'all': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply, tree.Value, tree.Variable ],
#     'functions': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply ],
#     'terminals': [ tree.Value, tree.Variable ]
# }

# rng = np.random.RandomState(seed=2)
# i = Individual(max_depth=7, n_features=4)
# i.grow(node_list, rng)
# print(i.tree)

# print('-----')

# i.full(node_list, rng)
# print(i.tree)