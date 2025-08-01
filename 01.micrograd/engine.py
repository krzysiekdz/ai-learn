from graphviz import Digraph
import matplotlib.pyplot as plt
import numpy as np

class Value:
    def __init__(self, data, _children=[], _op='', label = ''):
        self.data = data 
        self.grad = 0.0 # pochodna 0 oznacza brak wpływu na wyjscie 
        self._children = list(_children)
        self._op = _op
        self._backward = lambda : None
        self.label = label
    def __repr__(self) -> str :
        return f'Value({self.data}, op={self._op})'
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, [self, other], '+')
        def _backward():
            self.grad += out.grad  
            other.grad += out.grad
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other
    def __neg__(self):
        return self*(-1)
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data - other.data, [self, other], '-')
        def _backward():
            self.grad += out.grad  
            other.grad += out.grad
        out._backward = _backward
        return out
    def __rsub__(self, other):
        return self - other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, [self, other], '*')
        def _backward():
            self.grad += out.grad  * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out
    def __rmul__(self, other): 
        return self * other
    def __truediv__(self, other):
        return self * other**-1
    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, [self], f'pow({other})')
        def _backward():
            self.grad = out.grad * other * self.data**(other-1)
        out._backward = _backward
        return out
    def tanh(self):
        out = Value(np.tanh(self.data), [self], 'tanh')
        def _backward():
            self.grad += (1 - out.data**2) * out.grad
        out._backward = _backward
        return out
    def exp(self):
        out = Value(np.exp(self.data), [self], 'exp')
        def _backward():
            self.grad +=  out.data * out.grad  # d(e^x)/dx = e^x
        out._backward = _backward
        return out
    def backprop(self):
        self._backward()
        if(len(self._children) > 0):
            self._children[0].backprop()
            if(len(self._children) > 1):
                self._children[1].backprop()
    


def draw_graph(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'TB'})
    
    def build(parent, n, op):
        dot.node(name=str(id(n)), label='{%s|value=%.2f|grad=%.2f}' % (n.label, n.data, n.grad), shape='record')
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

# wykres funkcji tanh - tanges hiperboliczny - gwarantuje wartosc w przedziale (-1,1) - normalizacja
# plt.plot(x,  np.tanh(x) ); plt.grid()

# model neuronu:

# pryzkładowe wejscia 
x1 = Value(2, label='x1')
x2 = Value(-4, label='x2')
# wagi 
w1 = Value(3, label='w1') 
w2 = Value(5, label='w2')
# stała 
b = Value(15, label='b') 

x1w1 = x1 * w1 
x2w2 = x2 * w2 
x1w1x2w2 = x1w1 + x2w2 
n = x1w1x2w2 + b # 1.5
o = n.tanh() 

o.grad = 1
# reczne obliczenie, to wywolanie backward kolejno na elementach
# o._backward() 
# n._backward() 
# x1w1x2w2._backward() 
# b._backward() 
# x2w2._backward() 
# x1w1._backward()
# tutaj liczymy automatycznie , co jest równowanze powyzszym wywołaniom
# o.backprop()
# draw_graph(o)

# multivariable case :
a = Value(2, label='a')
b = Value(-3, label='b')
c = a + b ; c.label = 'c'
d = a * b ; d.label = 'd'
e = c + d ; e.label = 'e'
f = e * a ; f.label = 'f'
f.grad = 1
f.backprop()
# draw_graph(f)
# f = ((a + b) + a*b ) * a = a^2 + ab + a^2 b

# liczac np pochodna df/da klasycznie :
# df/da = 2a + b + 2ab = 4-3-12=-11 

# liczac z chain rule: 
# jesli f = z * a, gdzie z = (a+b+a*b)
# z = (-1 - 6) = -7 ;
# wszystkie zmienne "a" oznaczmy dla rozroznienia cyframi, czyli: 
# f = (a1 + b + a2 * b) * a3 , gdzie a1=a2=a3=a 
# df/da3 = (a1 + b + a2 * b) = -7 
# df/da1 = d (z * a3) / da1 = a3 * dz/da1 = a3 * d(a1 + b + a2 * b)/da1 = a3 * (1+0+0) = a3 = 2
# bo patrzymy tylko na zmienna a1 - dla niej liczymy 
# df/da2 = a3 * d(a1 + b + a2 * b)/da2 = a3 * (0+0+b) = a3 * b = -6 

# w sumie df/da to  : 
# df/da = df/da1 + df/da2 + df/da3 = z + a3 + a3*b = -7 + 2 -6 = -11 - wychodzi to samo, co klasycznie liczac
# jest to intuicyjne, bo zmienna w "roznych miejsach grafu" ma inny wplyw ("grad"), ktory sie sumuje
# czyli taka pochodną liczymy tak jakby a1,a2,a3 byly oddzielnymi zmiennymi, a nastepnie sumujemy te pochodne, bo jest to ta sama zmienna
# wiec zmienna "a" ma wplyw bedacy sumą pochodnych a1,a2,a3

# inny przyklad:
# f = x^2 * x^3 + 2x 
# df/dx = d(x^5 + 2x)/dx = 5x^4 + 2 - liczac klasycznie

# liczac z chain rule: 
# f = (x1)^2 * (x2)^3 + 2(x3)
# df/dx3 = 2 
# df/dx1 = (x2)^3 * 2(x1) = 2x^4 
# df/dx2 = (x1)^2 * 3(x2)^2 = 3x^4
# suma pochodnych = 2 + 2x^4 + 3x^4 = 5x^4 + 2 - wychodzi to samo jak liczac klasycznie

# z tej reguły wynika self.grad += ... zamiast self.grad = ... w funkcji _backward()

# takie operacje powinny byc wykonywalne
# a = Value(6) 
# # b = 4 * (1 + a)
# # c = a ** 2 
# b = (a / Value(3) ) ** 3
# b.grad = 1 
# b.backprop()
# draw_graph(b)

# takie operacje powinny byc wykonywalne
# a = Value(6)
# b = -a - Value(3)
# b.grad = 1 
# b.backprop()
# draw_graph(b)

#ostatecznie backpropagation działa dla modelu neuronu
o.backprop()
draw_graph(o)


