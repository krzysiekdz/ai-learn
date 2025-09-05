import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# makemore - podejscie z siecia neuronową

fname = 'imiona_polskie.txt'

def makemore(name, seed = 1234):
    # read data from file
    with open(name, encoding='utf-8') as f: 
        file_str = f.read()
    words = file_str.splitlines()
    words = [ w.lower() for w in words ]
    chars = sorted(list( set( ''.join(words) ) ))
    chars = ['.'] + chars
    chars_len = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    # prepare data set: x, y
    xs, ys = [], []
    for w in words[:1]:
        w = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(w, w[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            # print(f"{ch1}{ch2}")
            xs.append(idx1)
            ys.append(idx2)

    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    # print(xs, ys)
    xenc = F.one_hot(xs, num_classes=chars_len).float()
    # print(ys)
    # plt.imshow(xenc)
    # X = torch.tensor([[10.0, 20.0, 30.0, 40.0], [100, 200, 300, 400]])
    # W = torch.tensor([[1.0, 10], [2, 20], [3, 30], [4, 40]])
    # Y = X @ W + 5
    
    #tutaj dane są już gotowe, czyli: xenc, y 
    
    # przygotowanie wag 
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn((chars_len, chars_len), generator=gen)
    # print(W)
    # print(W[:, 2]) # neuron 2 - czyli kolumna 2 (liczone od 0)

    # forward pass:
    logits = xenc @ W # surowe dane z neuronów; dla xs = [0,1,4,1] logits to odpowiednio 0,1,4,1 wiersz W
    # czyli np W[0] to tak jakby prawdopodobienstwa co bedzie pierwsza litera (tak jak w bigramie)
    # W[i] to tak jakby prawdopodobienstwa jaki bedzie nastepny znak jesli poprzedni był [i]
    # kolumna to neuron - kazdy odpowiada za inny znak - mowi jakie jest prawdopodbienstwo tego znaku w zaleznosci co jest na wejsciu
    # czyli W to tak jakby bigram licznosci (ale nie dokładnie, patrz niżej)
    # softmax
    counts = torch.exp(logits) # to jest rownowazne N - dodatnie wartosci
    prob = counts / counts.sum(1, keepdim=True) # to jest rowówazne P
    # counts.sum(1, keepdim=True) = tensor([[29.1091],[50.8338],[38.9928],[50.8338]]) - przykładowe 4 sumy wierszy
    # print(prob)
    # print(prob[0].sum()) # = 1
    # prob - to analogicznie jak bigram - prawdopodobieństwa 
    # czyli na wyjsciu są prawdopodobienstwa - mozna z tego wyliczyc blad - jesli P bliskie 1 to blad bliski 0
    # np. blad dla pierszego wejscia to: print(prob[0, 1], prob[0, 1].log()) # 0 - pierwszy wiersz, 1 - odczytany z ys - prawidłowe wyjscie
    # loss = prob[ ,  ]
    loss = -prob[torch.arange(len(xenc)),  ys].log().mean()
    print(loss)
    # nalezy teraz obliczyc pochodna z wyrazenia na loss wzgledem kazdej wagi i zmodyfikowac wagi

        
makemore(fname)

# zaczac od backprop dla tej sieci
# wyliczenie bledu, minimalizowanie bledu -> backprop; modyfikowanie W - czyli przełozenie problemu na siec neuronową
# w ten sposob za pomoca sieci neuronowej mozna znalesc rozwiazanie problemu
# https://www.youtube.com/watch?v=PaCmpygFfXo&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=5




