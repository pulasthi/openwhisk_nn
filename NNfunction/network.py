"""
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#### Libraries
# Standard library
import random
import os
import subprocess

# Third-party libraries
import numpy as np
import couchdb
import json
import time
import requests

class Network(object):

    def __init__(self, sizes, dbname, rank, para, iterations):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.rank = rank
        self.para = para
        self.dbname = dbname
        self.iterations = iterations
        #load data from couchdb instead of generating them
        stime = time.time()*1000.0
        user = "whisk_admin"
        password = "some_passw0rd"
        self.couchserver = couchdb.Server("http://%s:%s@172.17.0.1:5984/" % (user, password))

        self.db = self.couchserver[self.dbname]
        wdoc = self.db.get('initw')
        bdoc = self.db.get('initb')
        self.biases = self.convertFromJSON(bdoc['b'])
        self.weights = self.convertFromJSON(wdoc['w'])
        etime = time.time()*1000.0
        print("{} - {}".format("time3", etime-stime))
        #self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        #self.weights = [np.random.randn(y, x)
        #                for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def convertFromJSON(self, data):
        convertedData = [];
        for x in data:
            convertedData.append(np.array(x))

        return convertedData

    def convertToJSON(self, data):
        convertedData = [];
        for x in data:
            convertedData.append(x.tolist())

        return convertedData

    def sum_and_convertToJSON(self, data1, data2):
        convertedData = [];
        for x , y in zip(data1, data2):
            convertedData.append(np.add(x,y).tolist())

        return convertedData

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        self.epochs = epochs
        self.eta = eta
        self.mini_batch_size = mini_batch_size
        if test_data: n_test = len(test_data)
        n = len(training_data)
        for j in xrange(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in xrange(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print "Epoch {0}: {1} / {2}".format(
                    j, self.evaluate(test_data), n_test)
            else:
                print "Epoch {0} complete".format(j)

        self.save_to_db()

    def save_to_db(self):
        lock = 'writeLock'
        functionCount = 'NNFunctionCount'
        global lockdoc
        global locked

        locked = False
        while(locked == False):
            lockdoc = self.db.get(lock)
            while (lockdoc['lock'] == 1):
                time.sleep(0.1)
                lockdoc = self.db.get(lock)

            lockdoc['lock'] = 1
            try:
                self.db.save(lockdoc)
            except couchdb.http.ResourceConflict:
                print('ResourceConflict')
                locked = False

            locked = True

        # we got the lock
        sumw = 'sumw'
        sumb = 'sumb'

        if sumb in self.db:
            sumwdoccr = self.db.get(sumw)
            sumbdoccr = self.db.get(sumb)
            biasescr = self.convertFromJSON(sumbdoccr['b'])
            weightscr = self.convertFromJSON(sumwdoccr['w'])
            sumwdoccr['w'] = self.sum_and_convertToJSON(weightscr, self.weights)
            sumbdoccr['b'] = self.sum_and_convertToJSON(biasescr, self.biases)
            self.db.save(sumwdoccr)
            self.db.save(sumbdoccr)
            print('sumwis there')

        else:
            sumwdoc = {'w': self.convertToJSON(self.weights)}
            sumbdoc = {'b': self.convertToJSON(self.biases)}
            self.db[sumw] = sumwdoc
            self.db[sumb] = sumbdoc
            print('sumw not there')

        #update function count
        funcCountdoc = self.db.get(functionCount)
        tempcount = funcCountdoc['count'] + 1
        funcCountdoc['count'] = tempcount
        self.db.save(funcCountdoc)

        #if this is the last function to update the call the invoker for the next round
        if(tempcount == self.para):
            actid = self.invokeNN()
            print(actid)

        lockdoc['lock'] = 0
        self.db.save(lockdoc)
        #released lock

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in xrange(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

    def invokeNN(self):
        iterCount = 'iterCount'
        iterdoc = self.db.get(iterCount)
        curriteration = iterdoc['count']
        #APIHOST = subprocess.check_output("wsk property get --apihost", shell=True).split()[2]
        #AUTH_KEY = subprocess.check_output("wsk property get --auth", shell=True).split()[2]
        #NAMESPACE = subprocess.check_output("wsk property get --namespace", shell=True).split()[2]
        NAMESPACE = os.environ.get('__OW_NAMESPACE')
        user_pass = os.environ.get('__OW_API_KEY').split(':')
        ACTION = 'nn1'
        PARAMS = {'dbname':self.dbname ,'layers': self.sizes, 'epochs': self.epochs,
         'eta':self.eta, 'mini_batch_size': self.mini_batch_size, 'para': self.para,
         'iter': self.iterations, 'curr': curriteration}
        BLOCKING = 'false'
        RESULT = 'true'
        APIHOST = 'http://172.17.0.1:8888'
        url = APIHOST + '/api/v1/namespaces/' + NAMESPACE + '/actions/' + ACTION

        response = requests.post(url, json=PARAMS, params={'blocking': BLOCKING, 'result': RESULT}, auth=(user_pass[0], user_pass[1]))
        return response.text

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))
