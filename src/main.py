import load
import train
import evaluate
import numpy as np

if __name__ == '__main__':
    train_features, train_labels, test_features, test_labels = load.execute()
    x, LRG = train.execute(train_features, train_labels)
    evaluate.execute(x, LRG, test_features, test_labels)