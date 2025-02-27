from graphviz import Digraph

class Value:
    def __init__(self, data, _children=[], _op='', _label = ''):
        self.data = data 
        self._children = list(_children)
        self._op = _op
        self._label = _label
    def __repr__(self) -> str :
        return f'Value({self.data}, op={self._op})'
    def __add__(self, other):
        return Value(self.data + other.data, [self, other], '+')
    def __sub__(self, other):
        return Value(self.data - other.data, [self, other], '-')
    def __mul__(self, other):
        return Value(self.data * other.data, [self, other], '*')
    


def draw_graph(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'TB'})
    
    def build(parent, n, op):
        dot.node(name=str(id(n)), label=f'{n.data}')
        if(op != ''):
            dot.node(name=str(id(parent))+op, label=f'{op}')
            dot.edge(str(id(n)), str(id(parent))+op)
        if(n._op != ''):
            dot.edge(str(id(n))+n._op, str(id(n)))
        if(len(n._children) > 0):
            build(n, n._children[0], n._op)
            if(len(n._children) > 1):
                build(n, n._children[1], n._op)
    
    build(None, root, '')
    
    return dot
    
# drzewo operacji z uzyciem klasy Value 
a = Value(4)
b = Value(-3)
c = Value(5)
e = a*b
d = e + c
f = d * Value(10)
print(f)
draw_graph(f)



