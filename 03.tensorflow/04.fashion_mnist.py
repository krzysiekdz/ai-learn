import tensorflow as tf
import keras
import numpy as np
import matplotlib.pyplot as plt

# load and prepare data 
fashion_mnist = keras.datasets.fashion_mnist
(train_img, train_labels), (test_img, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_img = train_img / 255.0 
test_img = test_img / 255.0

def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(), 
        keras.layers.Dense(16, activation='tanh'), 
        keras.layers.Dense(10)
    ])
    model.compile(optimizer='sgd', 
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
    
    return model 

def train_model(model, fname, e = 10):
    model.fit(train_img, train_labels, epochs=e)
    model.evaluate(train_img, train_labels, verbose=2)
    model.save_weights(fname)

def make_pred(model):
    pm = keras.Sequential([model, keras.layers.Softmax()])
    y = pm.predict(test_img)
    num_rows = 100
    num_cols = 2
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows,2*num_cols,2*i+1)
        plot_image(i, y[i], test_labels, test_img)
        plt.subplot(num_rows,2*num_cols,2*i+2)
        plot_value_array(i, y[i], test_labels)
    plt.tight_layout()

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.xticks([])
  plt.yticks([])
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
  
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')



# operacje
fname = './checkpoints/model_01'
model = create_model()
model.load_weights(fname)
make_pred(model)
# model.evaluate(train_img, train_labels, verbose=2)
# model.evaluate(test_img, test_labels, verbose=2)
# train_model(model, fname)




