import load
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import linear_model

#   use logistic regression to train model
#   gets dataset from load.py
#   returns trained model

def normal_equations_method(train_features, train_labels):
  A = np.array(train_features)
  b = np.array(train_labels)
  A_transpose = np.transpose(A)

  A_At = np.matmul(A_transpose, A)
  At_b = np.matmul(A_transpose, b)

  L = np.tril(A_At)
  U = np.triu(A_At)

  z = np.linalg.solve(L, At_b)
  x = np.linalg.solve(U, z)

  return x

def logistic_regression(train_features, train_labels):
  LRG = linear_model.LogisticRegression(
  random_state = 0,solver = 'liblinear',multi_class = 'auto'
  )
  A = np.array(train_features)
  b = np.array(train_labels)

  LRG.fit(A,b)
  return LRG

def execute(train_features, train_labels):
  x = normal_equations_method(train_features, train_labels)
  LRG = logistic_regression(train_features,train_labels)

  # Returns the x vector from normal equations, and logistic regression model

  return x, LRG



if __name__ == '__main__':
  execute(load.execute())
