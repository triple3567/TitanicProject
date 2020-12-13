import load
import train
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = load.execute()
    train.execute(train_features, train_labels, test_features, test_labels)