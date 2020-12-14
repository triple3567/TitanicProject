from sklearn import datasets
from sklearn import linear_model
from sklearn.metrics import plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def eval_lrg(LRG, test_features, test_labels):
    A = np.array(test_features)
    b = np.array(test_labels)

    print(LRG.score(A,b))

    solution = LRG.predict(A)

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for i in range(len(solution)):
        if solution[i] == b[i]:
            if solution[i] == 0:
                true_negative += 1
            else:
                true_positive += 1
        else:
            if solution[i] == 0:
                false_negative += 1
            else:
                false_positive += 1

    print(true_positive)
    print(true_negative)
    print(false_positive)
    print(false_negative)

    plot_confusion_matrix(LRG, A, b)
    plt.show()

def eval_normal_equation(x, test_features, test_labels):
    A = np.array(test_features)
    b = np.array(test_labels)

    solution = np.matmul(A,x)

    threshold = 0.001    #set a thershold for the solution
    for i in range(len(solution)):
        if abs(solution[i]) > threshold:
            solution[i] = 1
        else:
            solution[i] = 0
    
    print(solution)
    print(b)
    count = 0
    success = 0
    rate = 0
    for i in range(len(solution)):
        count += 1
        if solution[i] == b[i]:
            success += 1

def execute(x, LRG, test_features, test_labels):

    # evaluate LRG and plot
    eval_lrg(LRG, test_features, test_labels)
    
    # evaluate normal equations and plot
    eval_normal_equation(x, test_features, test_labels)

    