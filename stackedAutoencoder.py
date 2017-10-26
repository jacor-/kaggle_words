import dataReader
import new_autoencoders
from numpy import zeros, max, min
reload(new_autoencoders)


X_train, y_train = dataReader.readFile('dataset/wise2014-train.libsvm')

Y_train = zeros((X_train.shape[0], int(max(y_train)[0])))
for y_elem in range(len(y_train)):
    for value in y_train[y_elem]:
        Y_train[y_elem][int(value)-1] = 1




sa = new_autoencoders.SparseStandardAutoencoder(X_train.shape[1], 5000, N_classes = Y_train.shape[1])
sa.train(X_train, Y_train, 100, 0.001, 0.001)
