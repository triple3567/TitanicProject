from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import plot_roc_curve
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn

WRITE_PATH = "../data/predicted.csv"

def plot_confusion_matrix_normal(data, labels):
    """Plot confusion matrix using heatmap.
 
    Args:
        data (list of list): List of lists with confusion matrix data.
        labels (list): Labels which will be plotted across x and y axis.
        output_filename (str): Path to output file.
 
    """
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title("Normal Equations Method Confusion Matrix")
 
    seaborn.set(font_scale=1.4)
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
 
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
 
    ax.set(ylabel="True Label", xlabel="Predicted Label")

    plt.show()

def eval_lrg(LRG, test_features, test_labels):
    A = np.array(test_features)
    b = np.array(test_labels)

    solution = LRG.predict(A)
    score = LRG.score(A, b)

    # lrg header
    print()
    print("################################")
    print("#   Solving Titanic Problem    #")
    print("#  Using Logistic Regression   #")
    print("################################")

    # display info about model to console
    print()
    print("Accuracy of Logistic Regression Model on Test Set = ")
    print(score)

    # Coeficcient Values
    coef_temp = np.vstack([test_features.columns, LRG.coef_])
    coef = pd.DataFrame(coef_temp)
    print()
    print("Coefficient Values for Logistic Regression Model")
    print(coef)

    # plot confusion matrix
    plot_confusion_matrix(LRG, A, b)
    plt.title("Logistic Regression Confusion Matrix")
    plt.show()

    # plot roc curve
    plot_roc_curve(LRG, A, b)
    plt.title("Logistic Regression ROC curve")
    plt.show()

    # create solution csv
    predictions_csv = test_features.copy()
    predictions_csv.insert(12, "Predicted Label", np.transpose(solution))
    predictions_csv.insert(13, "Actual Label", test_labels)
    predictions_csv.to_csv(WRITE_PATH)

def eval_normal_equation(x, test_features, test_labels):
    A = np.array(test_features)
    b = np.array(test_labels)

    solution = np.matmul(A,x)

    # set a thershold for the solution
    # if a datapoint is greater that 0.5, its predicted they will survive
    # if the datapoint is less than 0.5, its predicted they will not survive
    threshold = 0.5    
    

    for i in range(len(solution)):
        if abs(solution[i]) > threshold:
            solution[i] = 1
        else:
            solution[i] = 0

    count = 0
    success = 0
    rate = 0
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    for i in range(len(solution)):
        count += 1
        if solution[i] == b[i]:
            success += 1
            if solution[i] == 1:
                true_pos += 1
            else:
                true_neg += 1
        else:
            if solution[i] == 1:
                false_pos += 1
            else:
                false_neg += 1

    rate = float(success) / float(count)

    x_temp = np.vstack([test_features.columns, x])
    x_label = pd.DataFrame(x_temp)

    print()
    print("################################")
    print("#   Solving Titanic Problem    #")
    print("# Using Normal Equation Method #")
    print("################################")

    print()
    print("Accuracy of Normal Equations Method on Test Set = ")
    print(rate)

    print()
    print("X solution vector for Normal Equations Method")
    print(x_label)

    # plot confustion matrix
    confustion_matrix_data = [[true_neg, false_pos],
                              [false_neg, true_pos]]
    labels = ['0', '1']    

    plot_confusion_matrix_normal(confustion_matrix_data, labels)

def execute(x, LRG, test_features, test_labels):

    pd.set_option('display.max_columns', None)
    # evaluate LRG and plot
    eval_lrg(LRG, test_features, test_labels)
    
    # evaluate normal equations and plot
    eval_normal_equation(x, test_features, test_labels)

    