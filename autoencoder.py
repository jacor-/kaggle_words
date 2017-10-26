
from theano.tensor.shared_randomstreams import RandomStreams
import theano
import theano.sparse
import theano.tensor as T
from layer import AEHiddenLayer
import numpy

from collections import OrderedDict

theano.config.warn.subtensor_merge_bug = False

class Nonlinearity:
    RELU = "rectifier"
    TANH = "tanh"
    SIGMOID = "sigmoid"

class CostType:
    MeanSquared = "MeanSquaredCost"
    CrossEntropy = "CrossEntropy"


class Autoencoder(object):

    def __init__(self,
            input,
            nvis,
            nhid=None,
            nvis_dec=None,
            nhid_dec=None,
            rnd=None,
            bhid=None,
            cost_type=CostType.MeanSquared,
            momentum=1,
            num_pieces=1,
            L2_reg=-1,
            L1_reg=-1,
            sparse_initialize=False,
            nonlinearity=Nonlinearity.TANH,
            bvis=None,
            tied_weights=True):

        self.input = input
        self.nvis = nvis
        self.nhid = nhid
        self.bhid = bhid
        self.bvis = bvis
        self.momentum = momentum
        self.nonlinearity = nonlinearity
        self.tied_weights = tied_weights
        self.gparams = None

        if cost_type == CostType.MeanSquared:
            self.cost_type = CostType.MeanSquared
        elif cost_type == CostType.CrossEntropy:
            self.cost_type = CostType.CrossEntropy

        if self.input is None:
            #self.input = theano.sparse.tensor.matrix('x')
            #self.input = theano.sparse.tensor.matrix('x')
            self.input = theano.sparse.csr_dmatrix('x')
        if rnd is None:
            self.rnd = numpy.random.RandomState(1231)
        else:
            self.rnd = rnd

        self.srng = RandomStreams(seed=1231)

        self.hidden = AEHiddenLayer(self.input,
                nvis,
                nhid,
                num_pieces=num_pieces,
                n_in_dec=nvis_dec,
                n_out_dec=nhid_dec,
                activation=None,
                tied_weights=tied_weights,
                sparse_initialize=sparse_initialize,
                rng=rnd)

        self.params = self.hidden.params

        self.L1_reg = L1_reg
        self.L2_reg = L2_reg

        self.sparse_initialize = sparse_initialize

        self.L1 = 0
        self.L2 = 0

        if L1_reg != -1:
            self.L1 += abs(self.hidden.W).sum()
            if not tied_weights and 0:
                self.L1 += abs(self.hidden.W_prime).sum()

        if L2_reg != -1:
            self.L2 += (self.hidden.W**2).sum()
            if not tied_weights and 0:
                self.L2 += (self.hidden.W_prime**2).sum()

        if self.input is not None:
            self.x = self.input
        else:
            #self.x = theano.sparse.tensor.matrix('x_input')
            #self.x = theano.sparse.tensor.matrix('x_input')
            self.x = theano.sparse.csr_dmatrix('x_input')

    def nonlinearity_fn(self, d_in=None, recons=False):
        if self.nonlinearity == Nonlinearity.SIGMOID:
            return T.nnet.sigmoid(d_in)
        elif self.nonlinearity == Nonlinearity.RELU and not recons:
            return T.maximum(d_in, 0)
        elif self.nonlinearity == Nonlinearity.RELU and recons:
            return T.nnet.softplus(d_in)
        elif self.nonlinearity == Nonlinearity.TANH:
            return T.tanh(d_in)

    def encode(self, x_in=None, center=True):
        if x_in is None:
            x_in = self.x
        print "---" + str(x_in.type)
        aux1 = theano.sparse.dot(x_in, self.hidden.W)
        aux2 = aux1 + self.hidden.b
        act = self.nonlinearity_fn(aux2)
        if center:
            act = act - act.mean(0)
        return act

    def decode(self, h):
        #return self.nonlinearity_fn(T.dot(h, self.hidden.W_prime) + self.hidden.b_prime)
        return theano.sparse.dot(h, self.hidden.W_prime)+self.hidden.b_prime

    def get_rec_cost(self, x_rec):
        """
        Returns the reconstruction cost.
        """
        if self.cost_type == CostType.MeanSquared:
            #return T.mean(((self.x - x_rec)**2).sum(axis=1))
            return theano.sparse.tensor.mean(((x_rec-self.x)**2).sum(axis=1)), theano.sparse.tensor.sum(((x_rec-self.x)**2).sum(axis=1))
        elif self.cost_type == CostType.CrossEntropy:
            return T.mean((T.nnet.binary_crossentropy(x_rec, self.x)).mean(axis=1)), T.sum((T.nnet.binary_crossentropy(x_rec, self.x)).mean(axis=1))

    def kl_divergence(self, p, p_hat):
        return p * T.log(p) - T.log(p_hat) + (1 - p) * T.log(1 - p) - (1 - p) * T.log(1 - p_hat)

    def sparsity_penalty(self, h, sparsity_level=0.05, sparse_reg=1e-3, batch_size=-1):
        if batch_size == -1 or batch_size == 0:
            raise Exception("Invalid batch_size!")
        sparsity_level = T.extra_ops.repeat(sparsity_level, self.nhid)
        sparsity_penalty = 0
        avg_act = h.mean(axis=0)
        kl_div = self.kl_divergence(sparsity_level, avg_act)
        sparsity_penalty = sparse_reg * kl_div.sum()
        # Implement KL divergence here.
        return sparsity_penalty

    def get_sgd_updates(self, learning_rate, lr_scaler=1.0, batch_size=-1, sparsity_level=-1, sparse_reg=-1, x_in=None):
        #JOSE: x_in is None, so the function encode assign self.x to this value, the tensor for the theano function
        h = self.encode(x_in)
        x_rec = self.decode(h)
        cost1, cost_total = self.get_rec_cost(x_rec)

        cost2 = 0
        if self.L1_reg != -1 and self.L1_reg is not None:
            cost2 += self.L1_reg * self.L1

        if self.L2_reg != -1 and self.L2_reg is not None:
            cost2 += self.L2_reg * self.L2

        if sparsity_level != -1 and sparse_reg != -1:
            sparsity_penal = self.sparsity_penalty(h, sparsity_level, sparse_reg, batch_size)
            cost2 += sparsity_penal

        self.gparams = theano.tensor.grad(cost1 + cost2, self.params)
        updates = OrderedDict({})
        for param, gparam in zip(self.params, self.gparams):
            updates[param] = self.momentum * param - lr_scaler * learning_rate * gparam
        return ([cost1+cost2, cost1, cost2, cost_total], updates)

    def fit(self,
            data=None,
            learning_rate=0.1,
            batch_size=100,
            n_epochs=20,
            lr_scaler=0.998,
            weights_file="out/ae_weights_mnist.npy"):
        """
        Fit the data to the autoencoder model. Basically this performs
        the learning.
        """
        if data is None:
            raise Exception("Data can't be empty.")

        #from theano.sparse import shared as sparse_shared

        index = T.lscalar('index')
        #data_shared = theano.shared(numpy.asarray(data.tolist(), dtype=theano.config.floatX))
        data_shared = theano.shared(data)
        n_batches = data.shape[0] / batch_size
        #JOSE: x_in must be none in order to fix this value as the tensor self.x assigned in the train function
        (cost, updates) = self.get_sgd_updates(learning_rate, lr_scaler, batch_size)
        
        #train_ae2 = theano.function([x_x],
        #                           self.decode(self.encode(x_in=x_x)),
        #                           )
        train_ae = theano.function([index],
                                   cost,
                                   updates=updates,
                                   givens={
                                       self.x: data_shared[index * batch_size: (index + 1) * batch_size]
                                       }
                                   )

        print "Started the training."
        ae_costs = []
        for epoch in xrange(n_epochs):
            aa_costs = []
            total_cost = 0
            #total_cost2 = 0
            print "Training at epoch %d" % epoch
            for batch_index in xrange(n_batches):
            
                #c2c2 = numpy.sum(  numpy.sum((data[batch_index*batch_size:(batch_index+1)*batch_size]-train_ae2(data[batch_index*batch_size:(batch_index+1)*batch_size]))**2,axis=1))
                #total_cost2 += c2c2
                #print "   --+-- " + str(c2c2)
                print str("batch : " + str(batch_index))           
                c1c1 = train_ae(batch_index)
                #print "   --+-- " + str(c1c1[-1])
                aa_costs.append(c1c1[:-1])
                total_cost+=c1c1[-1]
                
                
            ae_costs.append(aa_costs)
            a,b,c = numpy.mean(aa_costs, axis=0)
            print "Training at epoch %d, %f, %f, %f,   total: %f" % (epoch, a,b,c, total_cost / data.shape[0])
            #print " ------------------------ %f" % (total_cost2/len(data))

        #c3c3 = numpy.sum(  numpy.sum((data-train_ae2(data))**2,axis=1))
        #print "reconstruction value " + str(c3c3/len(data))
        
        print "Saving files..."
        #numpy.save(weights_file, self.params[0].get_value())
        return ae_costs


    def reconstruct(self, data_in):
        x_x = theano.sparse.csr_matrix('x_input')
        #x_x = theano.sparse.tensor.matrix('x_input')
        test_ae = theano.function([x_x], self.decode(self.encode(x_in=x_x)))
        return test_ae(data_in)

    def project(self, data_in):
        #x_x = theano.sparse.tensor.matrix('x_input')
        x_x = theano.sparse.csc_matrix('x_input')
        test_ae = theano.function([x_x], self.encode(x_in=x_x))
        return test_ae(data_in)

class AutoencoderLinearSigmoid(Autoencoder):
    def decode(self, h):
        #return self.nonlinearity_fn(T.dot(h, self.hidden.W_prime) + self.hidden.b_prime)
        return theano.tensor.dot(h, self.hidden.W_prime)+self.hidden.b_prime
 
class AutoencoderSigmoidSigmoid(Autoencoder):
    def decode(self, h):
        return self.nonlinearity_fn(theano.tensor.dot(h, self.hidden.W_prime) + self.hidden.b_prime)
        #return T.dot(h, self.hidden.W_prime)+self.hidden.b_prime




