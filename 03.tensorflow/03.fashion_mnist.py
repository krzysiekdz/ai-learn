import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()
# print(train_labels)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


train_img = train_img / 255.0 
test_img = test_img / 255.0
# plt.figure()
# plt.imshow(train_img[1])
# plt.colorbar()

# img = np.array([
#     [0,100,0,0],
#     [0,255,255,0],
#     [0,255,255,0],
#     [0,0,200,0],
# ])
# img = img / 255.0 

# wyswietlenie zdjec 5x5
# plt.figure(figsize=(10,10))
# for i in range(25,50):
#     plt.subplot(5,5,i+1-25)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(train_img[i], cmap=plt.cm.binary)
#     plt.xlabel(class_names[train_labels[i]])
# plt.show()

# przetestowanie layers.flatten - flatten powoduje, ze tablica wielowymiarowa staje sie jednowymiarowa
# input = tf.constant([
#     [[1.0,2,3],
#     [1.0,2,3],
#     [3.0,6,7],
#     ],
# ])
# l0 = keras.layers.Flatten()
# l1 = keras.layers.Dense(2)
# print(l0(input))

l0 = keras.layers.Flatten()
# print(l0(train_img)) # shape=(60000, 784), dtype=float32
l1 = keras.layers.Dense(12, activation='tanh')
# l2 = keras.layers.Dense(10, activation='tanh')
# out1 = l1(l0(train_img))
# print(out1) # shape=(60000, 64), dtype=float32)
l3 = keras.layers.Dense(10)
# out2 = l2(out1)
# print(out2) # shape=(60000, 10)

model = keras.Sequential()
model.add(l0)
model.add(l1)
# model.add(l2)
model.add(l3)
# print(model(train_img)) # shape=(60000, 10)
model.compile(optimizer='sgd', 
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
train_loss, train_acc = model.evaluate(train_img, train_labels, verbose=2)
model.summary()
# print(f'train losss = {train_loss}, train acc = {train_acc}')
model.fit(train_img, train_labels, epochs=4)

test_loss, test_acc = model.evaluate(test_img, test_labels, verbose=2)
# print(f'test losss = {test_loss}, test acc = {test_acc}')

# pmodel = keras.Sequential([model, keras.layers.Softmax()])
# pred = pmodel.predict(test_img)
# print(pred[0])



