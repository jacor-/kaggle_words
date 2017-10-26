

import csv
import numpy as np
import autoencoder
import dataReader
import sklearn

def readFile(filename):
    X_train, y_train = load_svmlight_file(filename, multilabel = True,dtype=np.float32)
    return X_train, y_train

X_train, y_train = dataReader.readFile('Data/wise2014-train.libsvm')
X_test, y_test = dataReader.readFile('Data/wise2014-test.libsvm')


classes = {}
for x in y_train:
	for x_k in x:
		if int(x_k) in classes:
			classes[ int(x_k)] +=1
		else:
			classes[int(x_k)] =1

print '\nnumber of classes',len(classes.keys())

missing = [ x for x in range(1,204) if x not in  classes.keys() ]

print '\nmissing classes in the train set',len(classes.keys())


y_train = LabelBinarizer().fit_transform(y_train)
y_test = LabelBinarizer().fit_transform(y_test)

datasumTrain = np.array(X_train.sum(axis = 0))[0] # sum all feature values of all the columns
(datasumTrain==0).sum() 
# 8920 columns are allways 0 -> Probably are usefull for classifying the classes
# that are not seen in the train set
zeroTrain = [feature_k for feature_k in range(datasumTrain.shape[0]) if datasumTrain[feature_k]==0]


datasumTest = np.array(X_test.sum(axis=0))[0]
sum(datasumTest)
# 449663.25 -> HI ha moltes més features al test no usades que al train
zeroTest = [feature_k for feature_k in range(datasumTest.shape[0]) if datasumTest[feature_k]==0]


## VOID INTERSECITON => IT is relevant the info of the feature sin zeroTrain in order
## to distinguish (and predict!!!) the classes missing classes in Train
set(zeroTrain).intersection(set(zeroTest))














