import nn

xs = [
      [2.0, 3.0, -1.0],
      [3.0, -1.0, 0.5],
      [0.5, 1.0, 1.0],
      [1.0, 1.0, -1.0],
   ]
ys = [1.0, -1.0, -1.0, 1.0]

def nn_train():
   n = nn.MLP(3, [3,3,1]) 
   n.calc_print(xs)
   n.train(xs, ys, 200, method=1) # method=1 daje bardzo dobre wyniki na poziomie 0.9999
   n.calc_print(xs)
   n.save_to_file('p.txt') # zapisanie parametrow w pliku


def nn_test():
   n = nn.MLP(filename='p.txt') 
   n.calc_print(xs)


# nn_train()
nn_test()
