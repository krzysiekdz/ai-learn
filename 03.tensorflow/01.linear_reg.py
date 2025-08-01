import tensorflow as tf
import matplotlib.pyplot as plt

# The actual line
TRUE_W = 3.0
TRUE_B = 2.0

NUM_EXAMPLES = 201

x = tf.linspace(-2, 2, NUM_EXAMPLES)
x = tf.cast(x, tf.float32)

def f(x):
    return x * TRUE_W + TRUE_B

noise = tf.random.normal([NUM_EXAMPLES])

y = f(x) + noise

# plt.plot(x, y)
# plt.show()

class LinearModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(1.0, name='w')
        self.b = tf.Variable(0.0, name='b')

    def __call__(self, x):
        return self.w * x + self.b

    def loss(self, y_pred, y_true):
        return tf.reduce_mean( tf.square(y_true-y_pred) )
    
    def train_step(self, x, y_true, step=0.001):
        with tf.GradientTape(persistent=True) as tape:
            loss = self.loss(self(x), y_true)
            dl_dw = tape.gradient(loss, self.w)
            dl_db = tape.gradient(loss, self.b)
            self.w.assign_sub(dl_dw * step)
            self.b.assign_sub(dl_db * step)

    def train(self, x, y_true, epochs=100, step=0.001, debug=False):    
        for i in range(epochs):
            self.train_step(x, y_true, step)
            if(debug):
                print('w', self.w)
                print('b', self.b)
        
    
model = LinearModel()
# y_pred = model(x)
# loss = model.loss(y_pred, y)
# print('loss = ', loss)
# print('Variables: ', model.variables)
# print('result = ', model(tf.constant(10)))
# plt.plot(x, y, '.', label='Data')
# plt.plot(x, f(x),  label='Ground Truth')
# plt.plot(x, y_pred, label='Predicate')
# plt.legend()

# przypomnienie gradientów
# a = tf.Variable([[1.,2,3]])
# w = tf.Variable([ [10.], [20], [30] ])
# b = tf.Variable([1.])

# with tf.GradientTape(persistent=True) as tape:
#     c = a @ w + b
#     print(c)
#     dc_dw = tape.gradient(c, w)
#     dc_db = tape.gradient(c, b)
#     print(dc_dw)
#     print(dc_db)
#     '''
#     tf.Tensor(
# [[1.]
#  [2.]
#  [3.]], shape=(3, 1), dtype=float32)
# tf.Tensor([1.], shape=(1,), dtype=float32)
# '''

model.train(x, y, debug=False, epochs=1000)
plt.plot(x, y, '.', label='Data')
plt.plot(x, f(x),  label='Ground Truth')
plt.plot(x, model(x), label='Predicate')
plt.legend()

print(f"w = ${model.w}, b = ${model.b}")
