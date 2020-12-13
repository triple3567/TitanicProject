import load
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import pandas as pd

#   use logistic regression to train model
#   gets dataset from load.py
#   returns trained model

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('Error [MPG]')
  plt.legend()
  plt.grid(True)
  plt.show()


def execute(train_features, train_labels, test_features, test_labels):

  print(train_features)
  print(train_labels)
  print(test_features)
  print(test_labels)

  



if __name__ == '__main__':
  execute(load.execute())
