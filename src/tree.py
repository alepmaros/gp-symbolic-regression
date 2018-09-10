class Node:
    def __init__(self):
        self.left  = None
        self.right = None
        self.value = None

    def eval(self):
        raise NotImplementedError

class Tree:
    def __init__(self):
        self.root = None

    def getRoot(self):
        return self.root

    def setRoot(self, root):
        self.root = root

    def evalTree(self):
        return self.root.eval()

    def __str__(self):
        if(self.root != None):
            return self.root.__str__()

class Function(Node):
    def __init__(self):
        self.type = 'Function'

class Sum(Function):
    def __init__(self):
        super().__init__()

    def eval(self):
        if (self.left == None or self.right == None):
            raise Exception('Left or Right value not set for Sum')
        
        return self.left.eval() + self.right.eval()

    def __str__(self):
        return '(+ {} {})'.format(self.left.__str__(), self.right.__str__())

class Terminal(Node):
    def __init__(self, val):
        super().__init__()
        self.value = val

    def eval(self):
        return self.value

    def __str__(self):
        return str(self.value)

a = Sum()
a.left = Sum()
a.left.left = Terminal(1)
a.left.right = Terminal(2)
a.right = Terminal(3)
t = Tree()
t.setRoot(a)
print(t.evalTree())
print(t)