import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

#
#   loads the training data
#   cleans the training data
#   formats the training data
#   normalized the training data
#   returns train_features, train_labels, test_features, test_labels
#

# Make numpy values easier to read.

BATCH_SIZE = 10
TRAIN_DATA_PATH = "../data/train.csv"

def execute():
    np.set_printoptions(precision=10, suppress=True)

    raw_dataset = pd.read_csv(TRAIN_DATA_PATH,
                              header=0,
                              skipinitialspace=True)
    dataset = raw_dataset.copy()
    
    #dropping cabin column because it contains most rows do not have a cabin value
    dataset = dataset.drop(columns='Cabin')

    #drop ticket number because its irelevant
    dataset = dataset.drop(columns='Ticket')

    #drop name because its irelevant
    dataset = dataset.drop(columns='Name')

    #drop passenger id because its irelevant
    dataset = dataset.drop(columns='PassengerId')

    #drop rows with no 'Embarked' value
    dataset = dataset[dataset['Embarked'].notna()]

    #set rows with no 'Age' value to Age='NaN'
    dataset.replace('',np.NaN)

    #replace 'Embarked' with one hot vectors
    dataset['Embarked'] = dataset['Embarked'].map(
        {'C': 'Cherbourg', 'Q': 'Queenstown', 'S': 'Southampton'}
    )

    #replace 'pclass' with one hot vectors
    dataset['Pclass'] = dataset['Pclass'].map(
        {1: 'Upper', 2: 'Middle', 3: 'Lower'}
    )

    #replace 'Sex' with one hot vectors
    dataset['Sex'] = dataset['Sex'].map(
        {'male': 'Male', 'female': 'Female'}
    )

    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')

    #split dataset into 80% training 20% testing
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    
    #split features and labels
    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop('Survived')
    test_labels = test_features.pop('Survived')

    #normalize features -- TODO
    #normalizer = preprocessing.Normalization()
    #normalizer.adapt(np.array(train_features))

    return train_features, train_labels, test_features, test_labels

if __name__ == '__main__':
    execute()