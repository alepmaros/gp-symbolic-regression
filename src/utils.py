import csv

def read_dataset(train_path, test_path):
    """
    Takes two paths for the train and test and return a list with the contents
    of the csv
    """
    train = []
    test = []

    with open(train_path, 'r') as csvfile:
        train = list(csv.reader(csvfile))
    
    with open(test_path, 'r') as csvfile:
        test = list(csv.reader(csvfile))
    
    return train, test