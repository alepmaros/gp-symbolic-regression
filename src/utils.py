import csv

import numpy as np

def read_dataset(train_path, test_path):
    """
    Takes two paths for the train and test and return a list with the contents
    of the csv
    """
    train = []
    test = []

    with open(train_path, 'r') as csvfile:
        train = np.array(list(csv.reader(csvfile)))
    
    with open(test_path, 'r') as csvfile:
        test = np.array(list(csv.reader(csvfile)))
    
    return train, test