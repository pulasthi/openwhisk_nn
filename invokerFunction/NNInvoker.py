import random

# Third-party libraries
import numpy as np
import couchdb
import json
"""
This function is used to invoke the NN functions
The parallizm that is needed is given to this function so that it will invoke the
NN functions as needed
"""
# convert lists that have numpy arrays to json serializable lists
def convertToJSON(data):
    convertedData = [];
    for x in data:
        convertedData.append(x.tolist())

    return convertedData

#converts list to have numpy arrays
def convertFromJSON(data):
    convertedData = [];
    for x in data:
        convertedData.append(np.array(x))

    return convertedData


class NNInvoker(object):

    def __init__(self, args):
        self.dbname = args.get("name", "digitnndb")
        self.iterations = args.get("iter", 10)
        self.parallelism = args.get("para", 4)
        self.currIter = args.get("curr", 0)
        self.sizes = args.get("layers", [784, 3, 10])

        #constants
        self.functionCount = 'NNFunctionCount'
        self.iterCount = 'iterCount'
        self.initw = 'initw'
        self.initb = 'initb'
        self.initnl = 'initnl'

        if(self.currIter == 0):
            #set up the database needed for this run, delete all data if exsists with same name
            user = "whisk_admin"
            password = "some_passw0rd"
            self.couchserver = couchdb.Server("http://%s:%s@172.17.0.1:5984/" % (user, password))

            if self.dbname in self.couchserver:
                del self.couchserver[self.dbname]

            self.db = self.couchserver.create(self.dbname)

            funccdoc = {'count': 0}
            iterdoc = {'count': 0}
            self.db[self.functionCount] = funccdoc
            self.db[self.iterCount] = iterdoc

            #If this is the first iteration init the weights and biases

            self.num_layers = len(self.sizes)
            self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
            self.weights = [np.random.randn(y, x)
                            for x, y in zip(self.sizes[:-1], self.sizes[1:])]

            wdoc = {'w': convertToJSON(self.weights)}
            bdoc = {'b': convertToJSON(self.biases)}
            nldoc = {'l': self.num_layers}

            self.db[self.initw] = wdoc
            self.db[self.initb] = bdoc
            self.db[self.initnl] = nldoc
            print('done')
        else:
            self.db = self.couchserver.create(dbname)
            doc1 = self.db.get(functionCount)
            print(doc1['count'])


        # save the initialized variables in couchdb

        #invoke the NN functions according to the parallizm

def main(args):
    nninvoker = NNInvoker(args)
    return {"LastDB": 'done'}
