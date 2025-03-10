from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np

class Value:
    def __init__(self, data, _children=[], _op='', label = ''):
        self.data = data 
        self.grad = 0.0 # pochodna 0 oznacza brak wpływu na wyjscie 
        self._children = list(_children)
        self._op = _op
        self.label = label
    def __repr__(self) -> str :
        return f'Value({self.data}, op={self._op})'
    def __add__(self, other):
        return Value(self.data + other.data, [self, other], '+')
    def __sub__(self, other):
        return Value(self.data - other.data, [self, other], '-')
    def __mul__(self, other):
        return Value(self.data * other.data, [self, other], '*')
    def tanh(self):
        return Value(np.tanh(self.data), [self], 'tanh')
    


def draw_graph(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'TB'})
    
    def build(parent, n, op):
        dot.node(name=str(id(n)), label='{%s|%.2f}' % (n.label, n.data), shape='record')
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
a = Value(4, label='a')
b = Value(-3, label='b')
c = Value(5, label='c')
d = a*b ; d.label = 'd'
e = d + c; e.label = 'e'
f = Value(10, label = 'f')
L = e * f; L.label = 'L' # loss function
L1 = L.data

h = 0.0001 #mała zmiana w celu obliczenia pochodnej wzgledem ktorejs ze zmiennych a,b,c,d,e,f

a = Value(4, label='a') # a = Value(4+h, label='a')
b = Value(-3, label='b') 
c = Value(5, label='c') 
d = a*b ; d.label = 'd' # d.data = d.data + h
e = d + c; e.label = 'e'
f = Value(10, label = 'f')
L = e * f; L.label = 'L' # loss function
L2 = L.data 

grad = (L2 - L1) / h
# print(grad)


#chain rule: jesli f(g(x)), to df|dx = df|dg * dg|dx
#korzystajac z chain rule mozna obliczyc:
a.grad = -30 # dL|da = -30
b.grad = 40 # dL|db = 40
c.grad = 10 # dL|dc = 10
# dL|dd = 10
# dL|de = 10
f.grad = -7 # dL|df = -7
# dL|dL = 1

# dodajemy do zmiennych wartosci pochodnych  - wartosc funkcji ulegnie zwiekszeniu
a.data += a.grad * 0.01
b.data += b.grad * 0.01
L = ((a*b) + c )*f
# print(L) # wartosc jest teraz wieksza, bo kazda zmienna tak wysterowalismy, aby zwiekszala wartosc funkcji

# draw_graph(L)

x = np.arange(-5,5, 0.1)
def f(x) :
    return 2*x 

# wykres funkcji tanh - tanges hiperboliczny
# plt.plot(x,  np.tanh(x) ); plt.grid()

# model neuronu:

# pryzkładowe wejscia 
x1 = Value(2, label='x1')
x2 = Value(-4, label='x2')
# wagi 
w1 = Value(3, label='w1') 
w2 = Value(5, label='w2')
# stała 
b = Value(15.5, label='b') 

x1w1 = x1 * w1 
x2w2 = x2 * w2 
x1w1x2w2 = x1w1 + x2w2 
o = x1w1x2w2 + b # 1.5
L = o.tanh()
print(L) # 0.91
draw_graph(L)
