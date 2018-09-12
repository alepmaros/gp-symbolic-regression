import numpy as np
import random

#########################
# BASE CLASSES          #
#########################

class Node:
    def __init__(self):
        self.left  = None
        self.right = None
        self.value = None
        self.parent = None

    def eval(self, X):
        raise NotImplementedError

class Tree:
    def __init__(self):
        self.root = None

    def getRoot(self):
        return self.root

    def setRoot(self, root):
        self.root = root

    def evalTree(self, X):
        return self.root.eval(X)

    def __str__(self):
        if(self.root != None):
            return self.root.__str__()

    def _getListOfNodes(self, node_list, node, position):
        if (node == None):
            return

        node_list.append( (position, node) )
        self._getListOfNodes(node_list, node.left, 'left')
        self._getListOfNodes(node_list, node.right, 'right')


    def getListOfNodes(self):
        node_list = []

        self._getListOfNodes(node_list, self.root.left, 'left')
        self._getListOfNodes(node_list, self.root.right, 'right')

        return node_list

#########################
# FUNCTIONS             #
#########################

class Function(Node):
    def __init__(self, parent, n_features):
        super().__init__()
        self.type = 'Function'
        self.n_features = n_features
        self.parent = parent

class Sum(Function):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features)

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Sum')
        
        return np.add(self.left.eval(X), self.right.eval(X))

    def __str__(self):
        return '(+ {} {})'.format(self.left.__str__(), self.right.__str__())

class Multiply(Function):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features)

    def eval(self, X):
        if (self.left == None or self.right == None):
            print(self.parent)
            raise Exception('Left or Right value not set for Multiply')
        
        return np.multiply(self.left.eval(X), self.right.eval(X))

    def __str__(self):
        return '(* {} {})'.format(self.left.__str__(), self.right.__str__())

class Subtraction(Function):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features)

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Subtraction')
        
        return np.subtract(self.left.eval(X), self.right.eval(X))

    def __str__(self):
        return '(- {} {})'.format(self.left.__str__(), self.right.__str__())
        
class Division(Function):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features)

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Division')
        
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(self.left.eval(X), self.right.eval(X))
            result[result == np.inf] = 0
            result = np.nan_to_num(result)
            return result

    def __str__(self):
        return '(/ {} {})'.format(self.left.__str__(), self.right.__str__())

#########################
# TERMINALS             #
#########################

class Terminal(Node):

    def __init__(self, parent, n_features):
        super().__init__()
        self.type = 'Terminal'
        self.n_features = n_features
        self.parent = parent

class Value(Terminal):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features)
        self.value = round(rng.uniform(-1, 1), 3)

    def eval(self, X):
        return np.repeat(self.value, X.shape[0])
        
    def __str__(self):
        return str(self.value)

class Variable(Terminal):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features)
        # Value here represents the i position of the X vector
        self.value = rng.randint(0, n_features)

    def eval(self, X):
        return X[:,self.value]

    def __str__(self):
        return 'X[{}]'.format(self.value)

# X = np.array([[1, 2, 3], [3, 4, 5], [6, 7, 8]])

# a = Sum()
# a.left = Multiply()
# a.left.left = Variable(0)
# a.left.right = Value(2)
# a.right = Variable(1)
# t = Tree()
# t.setRoot(a)
# print(t.evalTree(X))
# print(t)