from sklearn.datasets import load_svmlight_file

def readFile(filename):
    X_train, y_train = load_svmlight_file(filename, multilabel = True)
    return X_train, y_train


