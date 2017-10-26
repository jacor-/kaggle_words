# -*- coding: utf-8 -*-
import theano
import csv
import numpy as np
import autoencoder
import dataReader

import sklearn
from sklearn import svm
import sklearn.grid_search
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.multiclass import OneVsOneClassifier
from sklearn.datasets import make_multilabel_classification
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_svmlight_file
import numpy
from sklearn.datasets import fetch_20newsgroups

def to_utf8(X):
     #auxX  = []
     #for doc in X:
     #     auxX.append(doc.decode('utf-8', 'ignore'))
     #return auxX
     return X

def parse_data (data):
     # data has the following format
     # ['target_names', 'data', 'target', 'DESCR', 'filenames']
     new_data =[]
     no_Lines = [k for k,document_k in enumerate(data['data']) if 'Lines:' not in document_k ]
     # Do all the documents that do not contain 'Lines:'contain 'Organization:'?
     # Yes, because len(a)==len(no_Lines) where a is as follows:
     # a = [k for k in no_Lines if 'Organization:' in data['data'][k] ]
     for num,doc in enumerate(data['data']):
          if num in no_Lines:
               # Then document number num has Organization -> So take the information 
               # cutting from Organization: to the end of the file
               begin_doc = doc.index('Organization:') + 13
               new_doc = doc[begin_doc:len(doc)]
               new_data.append(new_doc)      
          else:
               # Cut in 'Lines:'
               begin_doc = doc.index('Lines:') + 6
               new_doc = doc[begin_doc:len(doc)]
               new_data.append(new_doc)
     return new_data

def clean_list_of_strings(data):
     '''
     data = list
     * List of documents
     '''
     clean_list = []
     #
     for doc in data:
          #doc = ''.join([i for i in doc.decode("utf-8") if ord(i) < 128])
          doc = unicode(doc)
          doc = doc.replace(u'.',' ')
          doc = doc.replace(u'-',' ')
          doc = doc.replace(u'@',' ')
          doc = doc.replace(u"_",' ')
          doc = doc.replace(u"#",' ')
          doc = doc.replace(u"/",' ')
          doc = doc.replace(u"\.",' ')
          doc = doc.replace(u"\x07",' ')
          doc = doc.replace(u"?",' ')
          doc = doc.replace(u"!",' ')
          doc = doc.replace(u"¿",' ')
          doc = doc.replace(u'"',' ')
          doc = doc.replace(u':',' ')
          doc = doc.replace(u',',' ')
          doc = doc.replace(u'>',' ')
          doc = doc.replace(u';',' ')
          doc = doc.replace(u'*',' ')
          doc = doc.replace(u')',' ')
          doc = doc.replace(u'(',' ')
          doc = doc.replace(u'<',' ')
          doc = doc.replace(u']',' ')
          doc = doc.replace(u'"',' ')
          doc = doc.replace(u"'",' ')
          doc = doc.replace(u"`",' ')
          doc = doc.replace(u"´",' ')
          doc = doc.replace(u'[',' ')
          doc = doc.replace(u'+',' ')
          doc = doc.replace(u'|',' ')
          doc = doc.replace(u'{',' ')
          doc = doc.replace(u'}',' ')
          doc = doc.replace(u'=',' ')
          clean_list.append(doc)
     return clean_list

def get_data_20news(numfeatures):
     Train = fetch_20newsgroups(subset='train')
     # Train.keys()
     # ['DESCR', 'data', 'target', 'target_names', 'filenames']
     Test = fetch_20newsgroups(subset='test')
     nTrain = []
     nTest = []
     nTrain =  parse_data (Train)
     nTest = parse_data(Test)
     print '\nTokenizing data\n'
     cleanTrain = clean_list_of_strings(nTrain)
     cleanTrain = to_utf8(cleanTrain)
     cleanTest = clean_list_of_strings(nTest)
     cleanTest = to_utf8(cleanTest)
     #
     cVectorizer = sklearn.feature_extraction.text.CountVectorizer(
     analyzer='word',
     #ngram_range = (1,3),
     max_features = numfeatures,
     min_df = 0.001,
     max_df = 0.95,
     )
     cVectorizer.fit(cleanTrain+cleanTest)
     # len(cVectorizer.vocabulary_)
     # cVectorizer.get_feature_names_()
     X_train = cVectorizer.transform(cleanTrain)
     X_train = X_train.toarray()
     # del cleanTrain
     X_test = cVectorizer.transform(cleanTest)
     X_test = X_test.toarray()
     # del cleanTest
     y_train = Train['target']
     y_test = Test['target']
     return X_train,X_test,y_train, y_test

def one_hot_encoding(y_train):
	y_train_def = numpy.zeros((len(y_train),max(y_train)))
	for i in range(len(y_train)):
		y_train_def[i][y_train[i]-1] = 1
	return np.array( y_train_def,dtype='float64')


class PoissonAutoencoder():
    '''
    This class implements the poisson autoencoder:
    Assumptions:
       We pass into the class count data 
       Count data is increased outsite the class by 1: (X_train + 1)
    '''
    def __init__(self, visible_size, hidden_size, ac, beta_rate, N_classes):
        #
        # Set data matrix
        self.X = theano.tensor.matrix("input data")
        # Set labels
        self.y = theano.tensor.matrix("label")
        # Set learning rate
        self.learning_rate = theano.tensor.dscalar()
        #
        initial_We = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(1. / (visible_size + hidden_size)),
                     high= numpy.sqrt(1. / (visible_size + hidden_size)),
                     size=(visible_size,hidden_size)))
        # theano shared variables for weights and biases
        We = theano.shared(value=initial_We, name='We')
        be = theano.shared(value=numpy.zeros(hidden_size), name='be')
        #
        initial_Wd = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(1. / (visible_size + hidden_size)),
                     high= numpy.sqrt(1. / (visible_size + hidden_size)),
                     size=(hidden_size, visible_size)))
        # theano shared variables for weights and biases
        Wd = theano.shared(value=initial_Wd, name='Wd')
        bd = theano.shared(value=numpy.zeros(visible_size), name='bd')
        #
        initial_Wc = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(1. / (visible_size + hidden_size)),
                     high= numpy.sqrt(1. / (visible_size + hidden_size)),
                     size=(hidden_size,N_classes)))
        # theano shared variables for weights and biases
        Wc = theano.shared(value=initial_Wc, name='Wc')
        bc = theano.shared(value=numpy.zeros(N_classes), name='bc')
        ## z = sigmoid ( log(X) * We  + be )
        z = theano.tensor.nnet.sigmoid(theano.tensor.dot(theano.tensor.log(self.X),We)+be)
        # Compute Er
        Er1 = beta_rate*theano.tensor.exp(theano.tensor.dot(z,Wd)+bd).sum(axis=1)
        Er2 = -theano.tensor.mul(self.X,theano.tensor.dot(z,Wd)).sum(axis=1)
        Er3 = -theano.tensor.dot(self.X,bd)
        Er = Er1 + Er2 + Er3
        #
        # softmax(vec) = np.exp(vec)/np.exp(vec).sum()
        h = theano.tensor.nnet.softmax(theano.tensor.dot(z,Wc)+bc)
        # Compute Ec
        # Ec = -sum_i( y_i * log(hi))
        Ec = -theano.tensor.mul(self.y,theano.tensor.log(h)).sum(axis=1)
        #
        # Cost we want to minimize. Notice that this cost is defined with a theano.tensor.mean
        # because it will be computed across examples in a batch
        #cost = theano.tensor.mean(Er + alpha_c * Ec)
        # If we use an online learning algorithm (batch_size=1) then the cost would be cost = Er + alpha_c * Ec
        cost = theano.tensor.mean(Er + ac * Ec)
        #
        # Parameters of the model
        params = [We,be,Wd,bd,Wc,bc]
        # Compute gradient of the cost with respect to the params
        grads = theano.tensor.grad(cost =cost, wrt = params)
        updates = [(param, param - self.learning_rate * grad) for grad,param in zip(grads, params)]
        # Update the needed
        self.updates = updates
        self.cost = cost
        self.params = params
        #
        X = theano.tensor.matrix()
        self.encode = theano.function([X], theano.tensor.nnet.sigmoid(theano.tensor.dot(theano.tensor.log(X),We)+be))
    #
    def encode(self, data):
        self.encode(data)
    #
    def train(self, train_data, labels, batch_size, learning_rate): 
        index = theano.tensor.lscalar()
        XX = theano.shared(train_data)
        YY = theano.shared(labels)
        f = theano.function(
                          [index],
                          self.cost,
                          updates=self.updates,
                          givens = { self.X: XX[index*batch_size:(index+1)*batch_size,:],
                                     self.learning_rate: learning_rate,
                                     self.y: YY[index*batch_size:(index+1)*batch_size,:]
                                   }
                          )
        batch_ = [f(i) for i in range(train_data.shape[0]/batch_size)]
        print str(numpy.mean(batch_))
        return batch_ 



def possible_batch_size(m):
     possible = []
     for x in range(1,m/2+1):
          if m%x ==0:
               possible.append(x)
     return possible


numfeatures = 2000
X_train,X_test,y_train,y_test = get_data_20news(numfeatures=numfeatures)

X_train = np.array( X_train +1.,dtype='float64')
X_test = np.array(X_test +1.,dtype='float64')

y_train_def = one_hot_encoding(y_train)

m =  X_train.shape[0]
possible = possible_batch_size(m)
batch_size = 2

beta_rate = 1.
visible_size = X_train.shape[1]
hidden_size = 1000
learning_rate = 0.001
ac = 0.01
N_classes = y_train_def.shape[1]

pa = PoissonAutoencoder(visible_size, hidden_size, ac, beta_rate, N_classes)

cost = pa.train(X_train, y_train_def, batch_size, learning_rate)



### FOR MORE EPOHCS:.....
costs = []
for epoch in range(10):
     #     costs += pa.train(X_train, y_train_def, batch_size, learning_rate/(1.+epoch))
    costs += pa.train(X_train, y_train_def, batch_size, learning_rate)
