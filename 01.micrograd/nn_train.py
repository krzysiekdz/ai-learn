import nn

xs = [
   [2.0, 3.0, -1.0],
   [3.0, -1.0, 0.5],
   [0.5, 1.0, 1.0],
   [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

n = nn.MLP(3, [4,4,1]) # MLP uzywa klasy Value, ktora tworzy drzewo operacji, aby potem latwo obliczyc pochodne - czyli backprop

# zadanie - wytrenowac siec, aby jej wagi w rezultacie dawaly porzadany wynik sieci
# na poczatku wagi sieci sa losowe 

# wyliczyc blad - funkcja bledu
# majac funkcje bledu - moge zmieniac wagi, aby minimalizowac blad

ypred = [ n(x) for x in xs ]
print([y.data for y in ypred])

def train(n_steps, train_step = 0.01):
   for s in range(1, n_steps+1):
      ypred = [ n(x) for x in xs ]
      yloss = [  (y-yp)**2 for yp, y in zip(ypred,ys)  ]
      loss = sum(yloss)
      print(f'step {s}, loss = {loss}')
      loss.grad = 1
      loss.backprop() # wyliczenie pochodnych - w jaki sposob waga ma wplyw na loss

      # print(n.parameters())
      # print(len(n.parameters())) # 41 parametrow 
      # nn.draw_graph(loss)

      # nalezy nastepnie wszystkie wagi zmodyfikowac korzystajac z wyliczoncyh pochodnych - tak, by wagi zmniejszały blad
      # potem nalezy wyzerowac gradient do kolejnej iteracji
      for p in n.parameters():
         p.data -=  train_step * p.grad
         p.grad = 0

train(200, 0.01)
ypred = [ n(x) for x in xs ]
print([y.data for y in ypred])