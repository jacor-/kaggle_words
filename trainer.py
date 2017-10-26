from numpy import array
import autoencoder
import dataReader
import new_autoencoder2
from numpy import zeros

X_train, y_train = dataReader.readFile('dataset/wise2014-train.libsvm')

Y_train = []
for y_elem in range(len(y_train)):
    Y_train.append([])
    for value in y_train[y_elem]:
        Y_train[-1].append(int(value)-1)


sa = new_autoencoder2.StandardAutoencoder(X_train.shape[1], 2, N_classes = int(max(y_train)[0]))

sa.train(X_train, array(Y_train), 2048, 0.01, 0.001)
