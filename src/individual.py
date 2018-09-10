import itertools, random
from src.tree import Tree

class Individual:
    def __init__(max_depth):
        self.tree = Tree()
        self.max_depth = max_depth

    def _grow(node, depth, available_nodes):
        if (node.type == 'function')
            r = random.randint(0, len(available_nodes)-1)
            root.value = available_nodes[r]()
            _grow(node, 1, available_nodes)


    def grow(available_nodes):
        
        root = self.tree.root

        r = random.randint(0, len(available_nodes)-1)
        root = available_nodes[r]()
        if (root.type == 'function'):
            _grow(node, 1, available_nodes)


    def full():
        print('full')
