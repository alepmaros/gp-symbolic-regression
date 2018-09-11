import itertools

import numpy as np
from src import tree

class Individual:
    def __init__(self, max_depth=7, n_features=2):
        self.tree       = tree.Tree()
        self.max_depth  = max_depth
        self.n_features = n_features

    def predict(self, X):
        y_pred = self.tree.evalTree(X)
        return y_pred
        
    #########################################
    # Methods of Creating New Individuals   #
    #########################################

    def _grow(self, node, depth, node_list, rgenerator):
        # If reached max depth, fill it with terminals
        if (depth >= self.max_depth):
            r = rgenerator.randint(0, len(node_list['terminals']))
            node = node_list['terminals'][r](self.n_features, rgenerator)
            return node

        # If still hasnt reached max depth, keep expanding it
        r = rgenerator.randint(0, len(node_list['all']))
        node = node_list['all'][r](self.n_features, rgenerator)
        if (node.type == 'Function'):
            node.left  = self._grow(node.left, depth+1, node_list, rgenerator)
            node.right = self._grow(node.right, depth+1, node_list, rgenerator)
        return node

    def grow(self, node_list, rgenerator):
        
        # Choose a random Node to be added to the tree
        r = rgenerator.randint(0, len(node_list['all']))

        # Set the root of the tree to be that node
        self.tree.root = node_list['all'][r](self.n_features, rgenerator)

        # If the node is a function, you still need to expand it
        if (self.tree.root.type == 'Function'):
            self.tree.root.left  = self._grow(self.tree.root.left, 1, node_list, rgenerator)
            self.tree.root.right = self._grow(self.tree.root.right, 1, node_list, rgenerator)

    def _full(self, node, depth, node_list, rgenerator):
        # If reached max depth, fill it with terminals
        if (depth >= self.max_depth):
            r = rgenerator.randint(0, len(node_list['terminals']))
            node = node_list['terminals'][r](self.n_features, rgenerator)
            return node

        # If still hasnt reached max depth, keep expanding with functions
        r = rgenerator.randint(0, len(node_list['functions']))
        node = node_list['functions'][r](self.n_features, rgenerator)
        node.left  = self._full(node.left, depth+1, node_list, rgenerator)
        node.right = self._full(node.right, depth+1, node_list, rgenerator)
        return node

    def full(self, node_list, rgenerator):
        # Choose a random Function to be added to the tree
        r = rgenerator.randint(0, len(node_list['functions']))
        self.tree.root = node_list['functions'][r](self.n_features, rgenerator)
        self.tree.root.left  = self._full(self.tree.root.left, 1, node_list, rgenerator)
        self.tree.root.right = self._full(self.tree.root.right, 1, node_list, rgenerator)


# node_list = {
#     'all': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply, tree.Value, tree.Variable ],
#     'functions': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply ],
#     'terminals': [ tree.Value, tree.Variable ]
# }

# rgenerator = np.random.RandomState(seed=2)
# i = Individual(max_depth=7, n_features=4)
# i.grow(node_list, rgenerator)
# print(i.tree)

# print('-----')

# i.full(node_list, rgenerator)
# print(i.tree)