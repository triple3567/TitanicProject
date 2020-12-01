import load
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#   use logistic regression to train model
#   gets dataset from load.py
#   returns trained model

def train():
  train_features, train_labels, test_features, test_labels = load.execute()

  print(train_features)
  print(train_labels)
  print(test_features)
  print(test_labels)


if __name__ == '__main__':
  train()
