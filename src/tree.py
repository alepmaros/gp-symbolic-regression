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

#########################
# FUNCTIONS             #
#########################

class Function(Node):
    def __init__(self):
        super().__init__()
        self.type = 'Function'

class Sum(Function):
    def __init__(self):
        super().__init__()

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Sum')
        
        return self.left.eval(X) + self.right.eval(X)

    def __str__(self):
        return '(+ {} {})'.format(self.left.__str__(), self.right.__str__())

class Multiply(Function):
    def __init__(self):
        super().__init__()

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Multiply')
        
        return self.left.eval(X) * self.right.eval(X)

    def __str__(self):
        return '(* {} {})'.format(self.left.__str__(), self.right.__str__())

class Subtraction(Function):
    def __init__(self):
        super().__init__()

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Subtraction')
        
        return self.left.eval(X) - self.right.eval(X)

    def __str__(self):
        return '(- {} {})'.format(self.left.__str__(), self.right.__str__())
        
class Division(Function):
    def __init__(self):
        super().__init__()

    def eval(self, X):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Division')
        
        return self.left.eval(X) / self.right.eval(X)

    def __str__(self):
        return '(/ {} {})'.format(self.left.__str__(), self.right.__str__())

#########################
# TERMINALS             #
#########################

class Terminal(Node):

    def __init__(self):
        super().__init__()
        self.type = 'Terminal'

class Value(Node):
    def __init__(self):
        super().__init__()
        self.value = random.uniform(-5, 5)

    def eval(self, X):
        return self.value

    def __str__(self):
        return str(self.value)

class Variable(Node):
    def __init__(self, val):
        super().__init__()

        # Value here represents the i position of the X vector
        self.value = random.randint(0, len(X[0])-2)

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