from copy import deepcopy
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

    def __deepcopy__(self, memodict={}):
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

    def _getListOfNodes(self, node_list, node, position, depth):
        if (node == None):
            return depth-1

        max_depth_left  = self._getListOfNodes(node_list, node.left, 'left', depth+1)
        max_depth_right = self._getListOfNodes(node_list, node.right, 'right', depth+1)
        max_depth = max([max_depth_left, max_depth_right])
        node_list.append({
            'position': position,
            'node': node,
            'cur_depth': depth,
            'node_depth': max_depth - depth})

        return max_depth

    def getListOfNodes(self):
        node_list = []

        self._getListOfNodes(node_list, self.root.left, 'left', 1)
        self._getListOfNodes(node_list, self.root.right, 'right', 1)

        return node_list

    def _getMaxDepth(self, node, depth):
        if (node == None):
            return depth-1
        
        max_depth_left  = self._getMaxDepth(node.left, depth+1)
        max_depth_right = self._getMaxDepth(node.right, depth+1)
        return max(max_depth_left, max_depth_right)

    def getMaxDepth(self):
        max_depth_left  = self._getMaxDepth(self.root.left, 1)
        max_depth_right = self._getMaxDepth(self.root.right, 1)

        return max(max_depth_left, max_depth_right)
        
    def __deepcopy__(self, memodict={}):
        copy_object = Tree()
        copy_object.root = deepcopy(self.root)
        return copy_object

#########################
# FUNCTIONS             #
#########################

class Function(Node):
    def __init__(self, parent, n_features, rng):
        super().__init__()
        self.type = 'Function'
        self.n_features = n_features
        self.parent = parent
        self.rng    = rng

class Sum(Function):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features, rng)

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Sum')
        
        return np.add(self.left.eval(X), self.right.eval(X))

    def __str__(self):
        return '({} + {})'.format(self.left.__str__(), self.right.__str__())

    def __deepcopy__(self, memodict={}):
        copy_object = Sum(None, self.n_features, self.rng)
        copy_object.parent       = None
        copy_object.left         = deepcopy(self.left)
        copy_object.right        = deepcopy(self.right)
        copy_object.left.parent  = copy_object
        copy_object.right.parent = copy_object
        return copy_object

class Multiply(Function):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features, rng)

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Multiply')
        
        return np.multiply(self.left.eval(X), self.right.eval(X))

    def __str__(self):
        return '({} * {})'.format(self.left.__str__(), self.right.__str__())

    def __deepcopy__(self, memodict={}):
        copy_object = Multiply(None, self.n_features, self.rng)
        copy_object.parent       = None
        copy_object.left         = deepcopy(self.left)
        copy_object.right        = deepcopy(self.right)
        copy_object.left.parent  = copy_object
        copy_object.right.parent  = copy_object
        return copy_object

class Subtraction(Function):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features, rng)

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Subtraction')
        
        return np.subtract(self.left.eval(X), self.right.eval(X))

    def __str__(self):
        return '({} - {})'.format(self.left.__str__(), self.right.__str__())

    def __deepcopy__(self, memodict={}):
        copy_object = Subtraction(None, self.n_features, self.rng)
        copy_object.parent       = None
        copy_object.left         = deepcopy(self.left)
        copy_object.right        = deepcopy(self.right)
        copy_object.left.parent  = copy_object
        copy_object.right.parent = copy_object
        return copy_object
        
class Division(Function):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features, rng)

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Division')
        
        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.true_divide(self.left.eval(X), np.round(self.right.eval(X), 2))
            result[np.isinf(result)] = 1
            result[np.isnan(result)] = 1
            return result

    def __str__(self):
        return '({} / {})'.format(self.left.__str__(), self.right.__str__())

    def __deepcopy__(self, memodict={}):
        copy_object = Division(None, self.n_features, self.rng)
        copy_object.parent       = None
        copy_object.left         = deepcopy(self.left)
        copy_object.right        = deepcopy(self.right)
        copy_object.left.parent  = copy_object
        copy_object.right.parent = copy_object
        return copy_object

#########################
# TERMINALS             #
#########################

class Terminal(Node):

    def __init__(self, parent, n_features, rng):
        super().__init__()
        self.type = 'Terminal'
        self.n_features = n_features
        self.parent = parent
        self.rng    = rng

class Value(Terminal):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features, rng)
        self.value = round(rng.uniform(-2, 2), 2)

    def eval(self, X):
        return np.full(X.shape[0], self.value)
        
    def __str__(self):
        return str(self.value)

    def __deepcopy__(self, memodict={}):
        copy_object = Value(None, self.n_features, self.rng)
        copy_object.value        = self.value
        copy_object.parent       = None
        copy_object.left         = None
        copy_object.right        = None
        return copy_object

class Variable(Terminal):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features, rng)
        # Value here represents the i position of the X vector
        self.value = rng.randint(0, n_features)

    def eval(self, X):
        return X[:,self.value]

    def __str__(self):
        return 'X{}'.format(self.value)

    def __deepcopy__(self, memodict={}):
        copy_object = Variable(None, self.n_features, self.rng)
        copy_object.value        = self.value
        copy_object.parent       = None
        copy_object.left         = None
        copy_object.right        = None
        return copy_object

class Sin(Terminal):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features, rng)
        self.value = rng.randint(0, n_features)

    def eval(self, X):
        return np.sin(X[:,self.value])

    def __str__(self):
        return '(sin(X{}))'.format(self.value)

    def __deepcopy__(self, memodict={}):
        copy_object = Sin(None, self.n_features, self.rng)
        copy_object.value        = self.value
        copy_object.parent       = None
        copy_object.left         = None
        copy_object.right        = None
        return copy_object

class Cos(Terminal):
    def __init__(self, parent, n_features, rng):
        super().__init__(parent, n_features, rng)
        self.value = rng.randint(0, n_features)

    def eval(self, X):
        return np.cos(X[:,self.value])

    def __str__(self):
        return '(cos(X{}))'.format(self.value)

    def __deepcopy__(self, memodict={}):
        copy_object = Cos(None, self.n_features, self.rng)
        copy_object.value        = self.value
        copy_object.parent       = None
        copy_object.left         = None
        copy_object.right        = None
        return copy_object

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