import itertools

from copy import deepcopy
import numpy as np
from src import tree

# from utils import read_dataset

class Individual:
    def __init__(self, max_depth=7, n_features=2):
        self.tree       = tree.Tree()
        self.max_depth  = max_depth
        self.n_features = n_features
        self.fitness    = None

        # self.list_nodes = None

    def predict(self, X):
        y_pred = self.tree.evalTree(X)
        return y_pred

    def select_random_node(self, rng):
        # if (self.list_nodes == None):
        nodes = self.tree.getListOfNodes()
        return nodes[rng.randint(0, len(nodes))]

    def __deepcopy__(self, memodict={}):
        copy_object = Individual()
        copy_object.tree       = deepcopy(self.tree)
        copy_object.max_depth  = self.max_depth
        copy_object.n_features = self.n_features
        copy_object.fitness    = None
        return copy_object


    #########################################
    # Methods of Creating New Individuals   #
    #########################################

    def _grow(self, parent, node, depth, node_list, rng, max_depth):
        # If reached max depth, fill it with terminals
        if (depth >= max_depth):
            r = rng.randint(0, len(node_list['terminals']))
            node = node_list['terminals'][r](parent, self.n_features, rng)
            return node

        # If still hasnt reached max depth, keep expanding it
        r = rng.randint(0, len(node_list['all']))
        node = node_list['all'][r](parent, self.n_features, rng)
        if (node.type == 'Function'):
            node.left  = self._grow(node, node.left, depth+1, node_list, rng, max_depth)
            node.right = self._grow(node, node.right, depth+1, node_list, rng, max_depth)
        return node

    def grow(self, node_list, rng, depth, max_depth):
        if (depth >= max_depth):
            r = rng.randint(0, len(node_list['terminals']))
            self.tree.root = node_list['terminals'][r](None, self.n_features, rng)
            return

        # Choose a random Node to be added to the tree. The Root needs to be a function
        r = rng.randint(0, len(node_list['functions']))

        # Set the root of the tree to be that node
        self.tree.root = node_list['functions'][r](None, self.n_features, rng)

        self.tree.root.left  = self._grow(self.tree.root, self.tree.root.left, depth+1, node_list, rng, max_depth)
        self.tree.root.right = self._grow(self.tree.root, self.tree.root.right, depth+1, node_list, rng, max_depth)

    def _full(self, parent, node, depth, node_list, rng, max_depth):
        # If reached max depth, fill it with terminals
        if (depth >= max_depth):
            r = rng.randint(0, len(node_list['terminals']))
            node = node_list['terminals'][r](parent, self.n_features, rng)
            return node

        # If still hasnt reached max depth, keep expanding with functions
        r = rng.randint(0, len(node_list['functions']))
        node = node_list['functions'][r](parent, self.n_features, rng)
        node.left  = self._full(node, node.left, depth+1, node_list, rng, max_depth)
        node.right = self._full(node, node.right, depth+1, node_list, rng, max_depth)
        return node

    def full(self, node_list, rng, max_depth):
        # Choose a random Function to be added to the tree
        r = rng.randint(0, len(node_list['functions']))
        self.tree.root = node_list['functions'][r](None, self.n_features, rng)
        self.tree.root.left  = self._full(self.tree.root, self.tree.root.left, 1, node_list, rng, max_depth)
        self.tree.root.right = self._full(self.tree.root, self.tree.root.right, 1, node_list, rng, max_depth)


# synth1_train = 'datasets/synth1/synth1-train.csv'
# synth1_test  = 'datasets/synth1/synth1-test.csv'
# train, test = read_dataset(synth1_train, synth1_test)
# X_train = train[:, :-1]
# y_train = train[:,-1]
# X_test  = test[:, :-1]
# y_test  = test[:, -1]
# n_features     = len(X_train[0])

# print(X_train)
# print(y_train)

# node_list = {
#     'all': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply, tree.Value, tree.Variable ],
#     'functions': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply ],
#     'terminals': [ tree.Value, tree.Variable ]
# }

# rng = np.random.RandomState()
# i = Individual(max_depth=7, n_features=n_features)
# i.tree.root = tree.Multiply(None, n_features, rng)
# i.tree.root.left = tree.Multiply(i.tree.root, n_features, rng)
# i.tree.root.left.left = tree.Variable(i.tree.root.left, n_features, rng)
# i.tree.root.left.right = tree.Variable(i.tree.root.left, n_features, rng)

# i.tree.root.right = tree.Multiply(i.tree.root, n_features, rng)
# i.tree.root.right.left = tree.Variable(i.tree.root.right, n_features, rng)
# i.tree.root.right.right = tree.Variable(i.tree.root.right, n_features, rng)

# i.grow(node_list, rng)
# print('i1', i.tree)
# print(i.predict(X_train))
# print(y_train - i.predict(X_train))

# i2 = deepcopy(i)
# print('i2', i2.tree)
# rnode = i2.select_random_node(rng)
# print('rnode', rnode['node'])

# rnode_copy = deepcopy(rnode['node'])
# print(rnode_copy.parent)
# print('rnode_copy', rnode_copy)
# rnode_copy = tree.Variable(rnode_copy.parent, 3, rng)
# print('rnode_copy_changed', rnode_copy)

# print('i1', i.tree)
# print('i2', i2.tree)

