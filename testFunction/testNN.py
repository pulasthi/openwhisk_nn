import couchdb

# Third-party libraries
import numpy as np
import email
import smtplib

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#converts list to have numpy arrays
def convertFromJSON(data):
    convertedData = [];
    for x in data:
        convertedData.append(np.array(x))

    return convertedData

def feedforward(a):
    """Return the output of the network if ``a`` is input."""
    for b, w in zip(biases, weights):
        a = sigmoid(np.dot(w, a)+b)
    return a

def vectorized_result(j):
    """Return a 10-dimensional unit vector with a 1.0 in the jth
    position and zeroes elsewhere.  This is used to convert a digit
    (0...9) into a corresponding desired output from the neural
    network."""
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_test_data():
    global couchserver

    user = "whisk_admin"
    password = "some_passw0rd"
    couchserver = couchdb.Server("http://%s:%s@172.17.0.1:5984/" % (user, password))
    dbname= 'digitnndata'
    db = couchserver[dbname]

    testdoc = db.get('testdata')
    testdata = testdoc['data']
    test_data = (np.array(testdata[0]), np.array(testdata[1]))
    return test_data

def load_data_wrapper():
    t_d = load_test_data()
    test_inputs = [np.reshape(x, (784, 1)) for x in t_d[0]]
    test_data = zip(test_inputs, t_d[1])
    return test_data

def main(args):
    global weights
    global biases
    global couchserver

    dbname = args.get('dbname', 'digitnndb')
    finalResults = 'finalResults'

    test_data = load_data_wrapper()

    #load weights and biases
    db = couchserver[dbname]
    wdoc = db.get('initw')
    bdoc = db.get('initb')
    biases = convertFromJSON(bdoc['b'])
    weights = convertFromJSON(wdoc['w'])

    """Return the number of test inputs for which the neural
    network outputs the correct result. Note that the neural
    network's output is assumed to be the index of whichever
    neuron in the final layer has the highest activation."""
    test_results = [(np.argmax(feedforward(x)), y)
                    for (x, y) in test_data]
    results = sum(int(x == y) for (x, y) in test_results)
    n_test = len(test_data)
    resultsrt = "Final Results: {0} / {1}".format(results, n_test)
    resultdoc = {'result': resultsrt}
    db[finalResults] = resultdoc
    return {"result": resultsrt}
