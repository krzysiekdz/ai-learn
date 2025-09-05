import torch
import matplotlib.pyplot as plt

fname = 'imiona_polskie.txt'

def makemore(name, print_bigram=False, print_distribution=False, count=10, seed=1234):
    with open(name, encoding='utf-8') as f: 
        file_str = f.read()
    words = file_str.splitlines()
    words = [ w.lower() for w in words ]
    # words = [w for w in words if 'a' not in w and 'A' not in w]
    # print( min([len(w) for w in words]) ) # 3
    # print( max([len(w) for w in words]) ) # 14
    # print( [ w for w in words if len(w) > 8] ) # imiona dłuższe niż 8 znaków
    # print(len(words)) # 626

    chars = sorted(list( set( ''.join(words) ) ))
    chars = ['.'] + chars
    chars_len = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    N = torch.zeros(chars_len, chars_len, dtype=torch.int32)

    for w in words[:]:
        w = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(w, w[1:]):
            N[stoi[ch1], stoi[ch2]] += 1

    # kilka najczęstszych bigramów
    # (('a', '.'), 309)
    # (('n', 'a'), 113)
    # (('a', 'n'), 100)
    # (('a', 'r'), 77)
    # (('i', 'a'), 73)
    # (('l', 'i'), 73)
    # ..........
    # kilka najrzadszych bigramów
    # (('c', '.'), 1)
    # (('e', 'ń'), 1)
    # (('i', 'ń'), 1)
    # (('ł', 'd'), 1)
    # (('u', 'z'), 1)
    # (('g', 'f'), 1)
    # (('g', 'm'), 1)
    
    if print_bigram:
        plt.figure(figsize=(16, 16))
        plt.imshow(N, cmap='Blues')
        for i in range(chars_len):
            for j in range(chars_len):
                plt.text(j, i, itos[i] + itos[j] , ha='center', va='bottom', color='gray')
                plt.text(j, i, int(N[i, j]) , ha='center', va='top', color='gray')
        plt.axis('off')

    # P = torch.ones(chars_len, chars_len, dtype=torch.float64)
    # for i in range(chars_len):
    #     P[i] /= P[i].sum()
    # dla jednakowwych prawdopodobienstw otrzymamy smieci, np: 'ejtp.', 'żcjnhńicvcnpghrzóaśłzzroophzzdśpuśkśóesepywyódvsżl', 'ęsudfóbvasś.', 'emłknkśżzomdnzkfjrmcjblłjnhużvleyko.',
    
    P = (N+0).float() #N+1 jest po to, żeby w loss function nie było nieskonczonosci (czyli log(0))
    #N+1 powoduje jednak pogorszenie modelu, bo pewne bezsensowne pary teraz staja sie prawdopodobne - wiecej smieci
    P /= P.sum(1, keepdim=True)

    # wystarczy też ta jedna linijka: P = N / torch.sum(N, dim=1, keepdim=True)
    
    #albo wersja z petlą
    # P = torch.zeros(chars_len, chars_len, dtype=torch.float64)
    # for i in range(chars_len):
    #     P[i] = N[i] / N[i].sum()
    
    if print_distribution:
        plt.figure(figsize=(16, 16))
        plt.imshow(P, cmap='Blues')
        for i in range(chars_len):
            for j in range(chars_len):
                plt.text(j, i, itos[i] + itos[j] , ha='center', va='bottom', color='gray')
                plt.text(j, i, f'{P[i, j]:.3f}'  , ha='center', va='top', color='gray')
        plt.axis('off')

    

    g = torch.Generator().manual_seed(seed)
    res = []

    # generating words
    for i in range(count):
        w = []
        idx = 0
        for j in range(50):
            idx = torch.multinomial(P[idx], num_samples=1, replacement=True, generator=g)
            idx = idx.item()
            if idx != 0:
                w.append(itos[idx])
            if idx == 0:
                break
        res.append(''.join(w))

    #loss function - wyliczenie bledu dla wygenerowanych słów
    loss = 0
    sum_loss = 0

    # for w in words: #wyliczenie bledu dla imion treningowych = 2.243
    # for w in ['sldkfjjjjjjjjjjjskddhsghggggggg']: #loss = 8.323
    for w in res:
        w = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(w, w[1:]):
            prob = P[stoi[ch1], stoi[ch2]]
            neg_log = -torch.log(prob)
            if neg_log > 9: 
                neg_log = 9
            # print("{0}{1} : {2}".format(ch1, ch2, prob))
            # print(f"{ch1}{ch2}: {neg_log:.2f} ({prob:.2f})")
            loss += neg_log
            sum_loss += 1
    
    loss /= sum_loss 
    print(f"loss = {loss}")

    return res

        
words = makemore(fname, print_bigram=False, print_distribution=True, count=100, seed=9877)
# print(words)
#dlaczego suma pionowo i poziomo jest taka sama?
# print(torch.sum(words, dim=0, keepdim=True, ))

# log(a*b*c) = log(a) + log(b) + log(c)
# loss - funkcja powinna zwracac małe wartosci, jesli jest mały blad oraz duze, jesli jest duzy
# https://www.wikiwand.com/en/articles/Maximum_likelihood_estimation
# mle - to przemnozenie wszystkich prawdopodobienstw (najlepiej, zeby bylo jak najblizsze 1)
# czyli to suma logarytmow - chcemy zminimalizowac tą sumę (w zalożeniu ze bierzemy wartosc bezwzgledna)
# model parameters - to bigram - czyli zliczone wystapienia par
# sieć neuronowa bedzie wyliczac te parametry - bigram
# cel treningu to zminimalizowanie loss 
# model smoothing - (N+1) - wszedzie w bigramie dodajemy "1" -> wtedy ln(x) nie moze byc infinite
# jesli np N+100 to model bedzie bardziej wygladzony


# https://pytorch.org/docs/stable/notes/broadcasting.html
# https://pytorch.org/docs/stable/generated/torch.sum.html


