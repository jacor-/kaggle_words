
import numpy
import theano

from numpy import zeros, array, ones
import theano.sparse

class SparseStandardAutoencoder():

    def __init__(self, visible_size, hidden_size, N_classes=2):

        self.X = theano.sparse.csr_matrix("input data")
        self.y = theano.tensor.matrix("label")
        
        initial_We = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     high= numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     size=(visible_size,hidden_size)))
        # theano shared variables for weights and biases
        We = theano.shared(value=initial_We, name='We')
        be = theano.shared(value=numpy.zeros(hidden_size), name='be')
        
        initial_Wd = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     high= numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     size=(hidden_size, visible_size)))
        # theano shared variables for weights and biases
        Wd = theano.shared(value=initial_Wd, name='Wd')
        bd = theano.shared(value=numpy.zeros(visible_size), name='bd')
        
        initial_Wc = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(1. / (visible_size + hidden_size)),
                     high= numpy.sqrt(1. / (visible_size + hidden_size)),
                     size=(hidden_size,N_classes)))
        # theano shared variables for weights and biases
        Wc = theano.shared(value=initial_Wc, name='Wc')
        bc = theano.shared(value=numpy.zeros(N_classes), name='bc')
                        
        z = theano.tensor.nnet.sigmoid(theano.sparse.dot(self.X,We)+be)
        recs = theano.tensor.dot(z,Wd)+bd

        X = theano.sparse.csr_matrix()
        self.project = theano.function([X], z, givens = {self.X: X})
        self.reconstruct = theano.function([X], recs, givens = {self.X: X})
        
        

        Er_rec = (theano.sparse.basic.sub(self.X, recs)**2).sum(axis=1)
        h = theano.tensor.nnet.softmax(theano.tensor.dot(z,Wc)+bc)
        Er_clas = theano.tensor.sum(-theano.tensor.mul(self.y, theano.tensor.log(h)),axis=1)

        self.learning_rate = theano.tensor.dscalar()        
        XX = theano.sparse.csr_matrix()
        YY = theano.tensor.matrix()
        index = theano.tensor.lscalar()
        batch_size = theano.tensor.lscalar()
        learning_rate = theano.tensor.dscalar()
        self.a_rate = theano.tensor.dscalar()
        a_rate = theano.tensor.dscalar()


        cost = theano.tensor.mean(Er_rec + self.a_rate * Er_clas)
        params = [We,be,Wd,bd,Wc,bc]
        grads = theano.tensor.grad(cost, params)
        updates = [(param, param - self.learning_rate * grad) for grad,param in zip(grads, params)]
        self.f_train = theano.function([XX, YY, index, batch_size, learning_rate, a_rate], cost, updates=updates, givens = {self.a_rate: a_rate, self.X: XX[index*batch_size:(index+1)*batch_size,:], self.learning_rate: learning_rate, self.y: YY[index*batch_size:(index+1)*batch_size,:]})
        self.f_monit = theano.function([XX, YY], [Er_rec, Er_clas], givens = {self.X: XX, self.y: YY})
        self.get_classification = theano.function([XX], h, givens = {self.X: XX})
        
    def classify(self, data):
        return self.get_classification(data)
    
    def project(self, data):
        return self.project(data)
        
    def reconstruct(self,data):
        return self.reconstruct(data)
    
    def train(self, train_data, labels, batch_size, learning_rate, a_rate): 
        for i in range(train_data.shape[0]/batch_size):
            print "batch: " + str(i) + " of " + str(train_data.shape[0]/batch_size) + "   " + str(self.f_train(train_data,labels,i,batch_size,learning_rate, a_rate))
        return


class StandardAutoencoder():

    def __init__(self, visible_size, hidden_size, N_classes=2):

        self.X = theano.tensor.matrix("input data")
        self.y = theano.tensor.matrix("label")
        
        initial_We = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     high= numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     size=(visible_size,hidden_size)))
        # theano shared variables for weights and biases
        We = theano.shared(value=initial_We, name='We')
        be = theano.shared(value=numpy.zeros(hidden_size), name='be')
        
        initial_Wd = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     high= numpy.sqrt(0.1 / (visible_size + hidden_size)),
                     size=(hidden_size, visible_size)))
        # theano shared variables for weights and biases
        Wd = theano.shared(value=initial_Wd, name='Wd')
        bd = theano.shared(value=numpy.zeros(visible_size), name='bd')
        
        initial_Wc = numpy.asarray(numpy.random.uniform(
                     low=- numpy.sqrt(1. / (visible_size + hidden_size)),
                     high= numpy.sqrt(1. / (visible_size + hidden_size)),
                     size=(hidden_size,N_classes)))
        # theano shared variables for weights and biases
        Wc = theano.shared(value=initial_Wc, name='Wc')
        bc = theano.shared(value=numpy.zeros(N_classes), name='bc')
                        
        z = theano.tensor.nnet.sigmoid(theano.tensor.dot(self.X,We)+be)
        recs = theano.tensor.dot(z,Wd)+bd

        X = theano.tensor.matrix()
        self.project = theano.function([X], z, givens = {self.X: X})
        self.reconstruct = theano.function([X], recs, givens = {self.X: X})
        
        

        Er_rec = ((self.X-recs)**2).sum(axis=1)
        h = theano.tensor.nnet.softmax(theano.tensor.dot(z,Wc)+bc)
        Er_clas = theano.tensor.sum(-theano.tensor.mul(self.y, theano.tensor.log(h)),axis=1)

        self.learning_rate = theano.tensor.dscalar()        
        XX = theano.tensor.matrix()
        YY = theano.tensor.matrix()
        index = theano.tensor.lscalar()
        batch_size = theano.tensor.lscalar()
        learning_rate = theano.tensor.dscalar()
        self.a_rate = theano.tensor.dscalar()
        a_rate = theano.tensor.dscalar()


        cost = theano.tensor.mean(Er_rec + self.a_rate * Er_clas)
        params = [We,be,Wd,bd,Wc,bc]
        grads = theano.tensor.grad(cost, params)
        updates = [(param, param - self.learning_rate * grad) for grad,param in zip(grads, params)]
        self.f_train = theano.function([XX, YY, index, batch_size, learning_rate, a_rate], cost, updates=updates, givens = {self.a_rate: a_rate, self.X: XX[index*batch_size:(index+1)*batch_size,:], self.learning_rate: learning_rate, self.y: YY[index*batch_size:(index+1)*batch_size,:]})
        self.f_monit = theano.function([XX, YY], [Er_rec, Er_clas], givens = {self.X: XX, self.y: YY})
        self.get_classification = theano.function([XX], h, givens = {self.X: XX})
        
    def classify(self, data):
        return self.get_classification(data)
    
    def project(self, data):
        return self.project(data)
        
    def reconstruct(self,data):
        return self.reconstruct(data)
    
    def train(self, train_data, labels, batch_size, learning_rate, a_rate): 
        for i in range(train_data.shape[0]/batch_size):
            print "batch: " + str(i) + " of " + str(train_data.shape[0]/batch_size) + "   " + str(self.f_train(train_data,labels,i,batch_size,learning_rate, a_rate))
        return
