import itertools, random

import tree

class Individual:
    def __init__(self, max_depth=7, n_features=2):
        self.tree       = tree.Tree()
        self.max_depth  = max_depth
        self.n_features = n_features

    #########################################
    # Methods of Creating New Individuals   #
    #########################################

    def _grow(self, node, depth, node_list):
        # If reached max depth, fill it with terminals
        if (depth >= self.max_depth):
            r = random.randint(0, len(node_list['terminals'])-1)
            node = node_list['terminals'][r](self.n_features)
            return node

        # If still hasnt reached max depth, keep expanding it
        r = random.randint(0, len(node_list['all'])-1)
        node = node_list['all'][r](self.n_features)
        if (node.type == 'Function'):
            node.left  = self._grow(node.left, depth+1, node_list)
            node.right = self._grow(node.right, depth+1, node_list)
        return node

    def grow(self, node_list):
        
        # Choose a random Node to be added to the tree
        r = random.randint(0, len(node_list['all'])-1)

        # Set the root of the tree to be that node
        self.tree.root = node_list['all'][r](self.n_features)

        # If the node is a function, you still need to expand it
        if (self.tree.root.type == 'Function'):
            self.tree.root.left  = self._grow(self.tree.root.left, 1, node_list)
            self.tree.root.right = self._grow(self.tree.root.right, 1, node_list)

    def _full(self, node, depth, node_list):
        # If reached max depth, fill it with terminals
        if (depth >= self.max_depth):
            r = random.randint(0, len(node_list['terminals'])-1)
            node = node_list['terminals'][r](self.n_features)
            return node

        # If still hasnt reached max depth, keep expanding with functions
        r = random.randint(0, len(node_list['functions'])-1)
        node = node_list['functions'][r](self.n_features)
        node.left  = self._full(node.left, depth+1, node_list)
        node.right = self._full(node.right, depth+1, node_list)
        return node

    def full(self, node_list):
        # Choose a random Function to be added to the tree
        r = random.randint(0, len(node_list['functions'])-1)
        self.tree.root = node_list['functions'][r](self.n_features)
        self.tree.root.left  = self._full(self.tree.root.left, 1, node_list)
        self.tree.root.right = self._full(self.tree.root.right, 1, node_list)


node_list = {
    'all': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply, tree.Value, tree.Variable ],
    'functions': [ tree.Sum, tree.Division, tree.Subtraction, tree.Multiply ],
    'terminals': [ tree.Value, tree.Variable ]
}

# i = Individual(max_depth=7, n_features=4)
# i.grow(node_list)
# print(i.tree)

# print('-----')

# i.full(node_list)
# print(i.tree)