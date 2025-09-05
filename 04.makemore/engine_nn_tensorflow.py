import matplotlib.pyplot as plt
import tensorflow as tf
import keras

# makemore - podejscie z siecia neuronową z wykorzystaniem tensorflow

fname = 'imiona_polskie.txt'

class Makemore(tf.Module):
    def __init__(self, chars_len: int, seed: int = 1234, **kwargs):
        tf.random.set_seed(seed)
        self.w = tf.Variable(tf.random.normal(shape=(chars_len, chars_len)))
    def __call__(self, x):
        logits = tf.matmul(x, self.w)
        counts = tf.exp(logits)
        sum = tf.reduce_sum(counts, axis = 1, keepdims=True)
        prob = counts / sum
        return prob
    def prob(self):
        logits = self.w
        counts = tf.exp(logits)
        sum = tf.reduce_sum(counts, axis = 1, keepdims=True)
        prob = counts / sum
        return prob
    def loss(self, prob, ys):
        indicies = tf.stack([tf.range(len(ys)), ys], axis=1) # tworzy indeksy do pobrania z prob
        # [[0 1]
        # [1 4]
        # [2 1]
        # [3 0]],
        q = tf.gather_nd(prob, indicies) # pobiera z prob wartosci z podanych indeksow
        return -tf.reduce_mean(tf.math.log(q)) # cross entropy
    def train_step(self, x, y_true, step=0.01):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss(self(x), y_true)
            dl_dw = tape.gradient(loss, self.w)
            self.w.assign_sub(dl_dw * step)
    def train(self, x, y_true, epochs=100, step=0.01, debug=False):    
        for i in range(epochs):
            self.train_step(x, y_true, step)
            if(debug):
                print('w', self.w)
            

def makemore(name, count=1, seed = 1234, print_distribution=False):
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
    for w in words[:count]:
        w = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(w, w[1:]):
            idx1 = stoi[ch1]
            idx2 = stoi[ch2]
            # print(f"{ch1}{ch2}")
            xs.append(idx1)
            ys.append(idx2)

    xs = tf.constant(xs)
    ys = tf.constant(ys)
    xenc = tf.one_hot(xs, depth=chars_len)
    # xenc = tf.cast(xenc, dtype=tf.int16)
    # print(xenc)
    # [1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # [0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    # [0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]

    m = Makemore(chars_len=chars_len, seed=seed)
    m.train(xenc, ys, step=0.1, epochs=200)
    
    P = m.prob()

    if print_distribution:
        plt.figure(figsize=(16, 16))
        plt.imshow(P, cmap='Blues')
        for i in range(chars_len):
            for j in range(chars_len):
                plt.text(j, i, itos[i] + itos[j] , ha='center', va='bottom', color='gray')
                plt.text(j, i, f'{P[i, j]:.3f}'  , ha='center', va='top', color='gray')
        plt.axis('off')

# trenowanie sieci na imieniu "Ada" - siec powinna nauczyć się, że po 'a' najczęściej występuje 'd' lub '.' (koniec imienia)
# generowane imiona powinny być podobne do "Ada", np: 'Ala', 'Aga', 'Ania', 'Aśka', 'Aka', 'Ala.', 'Ada.', 'Alaa', 'Adaa', 'Adaa'
makemore(fname, count=1, print_distribution=True)

# dodac generowanie imion; sprawdzic dla innych imion jako danych wejsciowych
# moge zawrzec inne pliki jako dane wejsciowe ktore beda mialy tylko kilka imion i zobaczyc jak sie dziala 
# bedzie to pokazywac jak siec sie uczy i czy trenowanie jest prawidlowe  




