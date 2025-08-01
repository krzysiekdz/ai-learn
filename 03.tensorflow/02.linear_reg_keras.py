import tensorflow as tf
import keras
from keras import layers
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

class LinearModelKeras(keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.w = tf.Variable(1.0, name='w')
        self.b = tf.Variable(0.0, name='b')

    def call(self, x):
        return self.w * x + self.b


