import torch 

x1 = torch.Tensor([2]).double()         ;x1.requires_grad = True
x2 = torch.Tensor([-4]).double()        ;x2.requires_grad = True
w1 = torch.Tensor([3]).double()         ;w1.requires_grad = True
w2 = torch.Tensor([5]).double()         ;w2.requires_grad = True
b = torch.Tensor([15]).double()         ;b.requires_grad = True
n = x1*w1 + x2*w2 + b
o = n.tanh()
print('output = ', o.data.item())
o.backward()

print('x1 grad =', x1.grad.item())
print('x2 grad =', x2.grad.item())
print('w1 grad =', w1.grad.item())
print('w2 grad =', w2.grad.item())
print('b grad =', b.grad.item())

# wyniki sa takie same jak przy uzyciu wlasnej klasy do liczenia backpropagation :)
# rzutowanie na double() jest dlatego, ze python domyslnie uzywa podwojnej precyzji, 
# cyzli float64 - tak jak w mojej klasie, a pytorch uzywa float32, dlatego zeby wyniki byly takie same robie rzutowanie
# pytorch domyslnie robi requires_grad = false ze wzgledow wydajnosciowych