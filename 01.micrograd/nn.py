from engine import Value, draw_graph
import numpy as np


class Neuron: 
    def __init__(self, n, type = 0):
        self.w = [ Value(i) for i in np.random.uniform(-1,1,n) ]
        self.b = Value(np.random.uniform(-1,1))
        self.type = type
    def __repr__(self):
        return f'weights = {self.w}, bias = {self.b}'
    def __call__(self, x):
        y = np.matmul(x, self.w) + self.b
        return y.tanh()
        # y =  sum((xi*wi for xi, wi in zip(x, self.w)), self.b)
        # if self.type == 0:
        #     return y if y.data >= 0 else Value(0)
        # else : 
        #     return y.tanh()
        # return y if y.data >= 0 else Value(0)
    def parameters(self):
        return self.w + [self.b]
    def init_from_params(self, params):
        for w in self.w :
            w.data = params.pop(0)
        self.b.data = params.pop(0)
    
class Layer:
    def __init__(self, n_in, n_out, type = 0):
        self.n = [Neuron(n_in, type) for _ in range(n_out)]
        self.n_in = n_in 
        self.n_out = n_out
    def __call__(self, x):
        out = [n(x) for n in self.n]
        return out
    def __repr__(self):
        return f'(n_in = {self.n_in}, n_out = {self.n_out} )'
    def parameters(self):
        return [ p for n in self.n for p in n.parameters() ]
    def init_from_params(self, params):
        for n in self.n :
            n.init_from_params(params)
    def set_type(self, type):
        for n in self.n: 
            n.type = type

    
class MLP:
    def __init__(self, n_in = 1, n_out = [1], filename = None):
        if filename is not None : 
            self.from_file(filename) 
        else :
            self.n = [n_in] + n_out
            n = self.n
            self.layers = [ Layer(n[i], n[i+1], type = 0) for i in range(len(n)-1)  ]
            lay_count = len(self.layers)
            self.layers[lay_count-1].set_type(1)
    def from_file(self, filename):
        numbers = []
        sizes = []
        try:
            with open(filename, 'r') as file:
                content = file.read().strip() # Read all content and remove leading/trailing whitespace
                if not content:
                    print(f"File '{filename}' is empty or contains only whitespace.")
                    return

                # Split the content by the specified separator
                # Filter out any empty strings that might result from splitting (e.g., extra newlines)
                num_strings = [s for s in content.split('\n') if s.strip()]

                nn_sizes = num_strings.pop(0)
                sizes = [int(s) for s in nn_sizes.split(' ') if s.strip()]

                for s in num_strings:
                    try:
                        # Try converting to int first, then float if int fails (for numbers like "30.5")
                        if '.' in s: # Simple check for potential float
                            numbers.append(float(s))
                        else:
                            numbers.append(int(s))
                    except ValueError:
                        print(f"Warning: Could not convert '{s}' to a number. Skipping.")
            print(f"Successfully read Neural network from '{filename}'.")
            self.init_from_params(sizes, numbers)
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
    def init_from_params(self, sizes, params):
        self.n = sizes
        n = self.n
        self.layers = [ Layer(n[i], n[i+1], type = 0) for i in range(len(n)-1)  ]
        lay_count = len(self.layers)
        self.layers[lay_count-1].set_type(1)
        for lay in self.layers :
            lay.init_from_params(params)
    def __call__(self, x):
        for lay in self.layers: 
            x = lay(x)
        return x[0] if len(x) == 1 else x
    def parameters(self):
        return [ p for lay in self.layers for p in lay.parameters() ]
    
    def train(self, xs, ys, n_steps, method = 0, train_step = 0.01, log = False):
        for s in range(1, n_steps+1):
            ypred = [ self(x) for x in xs ]
            yloss = [  (y-yp)**2 for yp, y in zip(ypred,ys)  ]
            loss = sum(yloss)
            if log : print(f'step {s}, loss = {loss}')
            loss.grad = 1
            loss.backprop() # wyliczenie pochodnych - w jaki sposob waga ma wplyw na loss
            # nalezy nastepnie wszystkie wagi zmodyfikowac korzystajac z wyliczoncyh pochodnych - tak, by wagi zmniejszały blad
            # potem nalezy wyzerowac gradient do kolejnej iteracji
            for p in self.parameters():
                if method == 0: 
                    p.data -=  train_step * p.grad
                if method == 1 : 
                    p.data -=  train_step if p.grad > 0 else -train_step
                p.grad = 0

    def calc(self, xs): 
        return [ self(x) for x in xs ]

    def calc_print(self, xs):
        ypred = self.calc(xs)
        print([y.data for y in ypred])

    def save_to_file(self, filename):
        p = [n.data for n in self.parameters()]
        try:
            with open(filename, 'w') as file:
                file.write(' '.join(map(str, self.n)))
                file.write('\n')
                for n in p:
                    file.write(str(n) + '\n')
                print(f"Neural network written to '{filename}'")
        except IOError as e:
            print(f"Error writing neural network to file '{filename}': {e}")

# przyklad uzycia klasy neuron
# n1 = Neuron(2)
# n2 = Neuron(2)
# x = [2, -3]
# o1 = n1(x)
# o2 = n2(x)

# print(o1)
# o1.grad = 1 
# o1.backprop()
# draw_graph(o1)

# x = [1,2,3]
# l1 = Layer(3,4)
# o1 = l1(x)
# print(o1)
# draw_graph(o1[0])
# o1[0] to pojedyczny neuron

# nn = MLP(3, [4,5,2, 1])
# print(nn.layers)
# o = nn(x)
# print(o)
# o.grad = 1 
# o.backprop()
# draw_graph(o)






