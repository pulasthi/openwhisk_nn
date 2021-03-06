import random
import os
import subprocess
import time

# Third-party libraries
import numpy as np
import couchdb
import json
import requests


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

def div_and_convertFromJSON(data, para):
    convertedData = [];
    for x in data:
        convertedData.append(np.divide(np.array(x),para))

    return convertedData

class NNInvoker(object):

    def __init__(self, args):


        self.dbname = args.get("name", "digitnndb")
        self.iterations = args.get("iter", 8) #save
        self.parallelism = args.get("para", 4)
        self.currIter = args.get("curr", 0)
        self.sizes = args.get("layers", [784, 15, 10])
        self.epochs = args.get("epochs", 1)
        self.eta = args.get("eta", 3.0)
        self.mini_batch_size = args.get("mini_batch_size", 10)
        #constants

        global complete
        complete = False
        self.functionCount = 'NNFunctionCount'
        self.iterCount = 'iterCount'
        self.lock = 'writeLock'
        self.initw = 'initw'
        self.sumw = 'sumw'
        self.initb = 'initb'
        self.sumb = 'sumb'
        self.initnl = 'initnl'

        user = "whisk_admin"
        password = "some_passw0rd"
        self.couchserver = couchdb.Server("http://%s:%s@149.165.150.85:30301/" % (user, password))

        if(self.currIter == 0):
            #set up the database needed for this run, delete all data if exsists with same name
            if self.dbname in self.couchserver:
                del self.couchserver[self.dbname]

            self.db = self.couchserver.create(self.dbname)

            funccdoc = {'count': 0}
            iterdoc = {'count': 1} # set the iteration number
            lockdoc = {'lock': 0}
            millis = int(round(time.time() * 1000))
            startTime = {'stime': millis}
            self.db['startTime'] = startTime
            self.db[self.functionCount] = funccdoc
            self.db[self.iterCount] = iterdoc
            self.db[self.lock] = lockdoc

            #If this is the first iteration init the weights and biases

            self.num_layers = len(self.sizes)

            #better method for weight initb
            tempdb = self.couchserver['initdatadb']
            wdoc = tempdb.get('constw')
            bdoc = tempdb.get('constb')
            nldoc = {'l': self.num_layers}


            self.db[self.initw] = wdoc
            self.db[self.initb] = bdoc
            self.db[self.initnl] = nldoc
            print('done')
            testre1 = self.invokeTest()
            time.sleep(10)
            print testre1
        else:
            self.db = self.couchserver[self.dbname]
            funcCountdoc = self.db.get(self.functionCount)
            iterdoc = self.db.get(self.iterCount)
            itercounttemp = iterdoc['count'] + 1

            iterdoc['count'] = itercounttemp
            self.db.save(iterdoc)
            sumwdoccr = self.db.get(self.sumw)
            sumbdoccr = self.db.get(self.sumb)
            biasescr = div_and_convertFromJSON(sumbdoccr['b'], self.parallelism)
            weightscr = div_and_convertFromJSON(sumwdoccr['w'], self.parallelism)
            print('after divide')

            wdoc = self.db.get(self.initw)
            bdoc = self.db.get(self.initb)

            wdoc['w'] = convertToJSON(weightscr)
            bdoc['b'] = convertToJSON(biasescr)
            self.db.save(wdoc)
            self.db.save(bdoc)

            #delete old sumw and wumb
            self.db.delete(sumwdoccr)
            self.db.delete(sumbdoccr)

            #reset counters
            funcCountdoc['count'] = 0
            self.db.save(funcCountdoc)
            lockdoc = self.db.get(self.lock)
            lockdoc['lock'] = 0
            self.db.save(lockdoc)


            if(itercounttemp > self.iterations):
                millis = int(round(time.time() * 1000))
                endTime = {'etime': millis}
                self.db['endTime'] = endTime
                complete = True
                print('invoke test function')
            else:
                #only needed for checking convergence
                testre = self.invokeTest()
                time.sleep(10)
                print testre


    def trainNN(self):
        #this method will run all the functions
        actIds = []
        for x in range(self.parallelism):
            actid = self.invokeNN(x)
            actIds.append(actid)

        return actIds

    def invokeNN(self, rank):
        #APIHOST = subprocess.check_output("wsk property get --apihost", shell=True).split()[2]
        #AUTH_KEY = subprocess.check_output("wsk property get --auth", shell=True).split()[2]
        #NAMESPACE = subprocess.check_output("wsk property get --namespace", shell=True).split()[2]
        NAMESPACE = os.environ.get('__OW_NAMESPACE')
        user_pass = os.environ.get('__OW_API_KEY').split(':')
        ACTION = 'digitnn'
        PARAMS = {'dbname':self.dbname ,'layers': self.sizes, 'epochs': self.epochs,
         'eta':self.eta, 'mini_batch_size': self.mini_batch_size, 'rank': rank, 'para': self.parallelism,
         'iter': self.iterations}
        BLOCKING = 'false'
        RESULT = 'true'
        APIHOST = 'http://172.17.0.1:8888'
        url = APIHOST + '/api/v1/namespaces/' + NAMESPACE + '/actions/' + ACTION

        response = requests.post(url, json=PARAMS, params={'blocking': BLOCKING, 'result': RESULT}, auth=(user_pass[0], user_pass[1]))
        return response.text

    def invokeTest(self):
        #APIHOST = subprocess.check_output("wsk property get --apihost", shell=True).split()[2]
        #AUTH_KEY = subprocess.check_output("wsk property get --auth", shell=True).split()[2]
        #NAMESPACE = subprocess.check_output("wsk property get --namespace", shell=True).split()[2]
        NAMESPACE = os.environ.get('__OW_NAMESPACE')
        user_pass = os.environ.get('__OW_API_KEY').split(':')
        ACTION = 'testnn'
        PARAMS = {'dbname':self.dbname ,'layers': self.sizes, 'epochs': self.epochs,
         'eta':self.eta, 'mini_batch_size': self.mini_batch_size, 'para': self.parallelism,
         'iter': self.iterations}
        BLOCKING = 'false'
        RESULT = 'true'
        APIHOST = 'http://172.17.0.1:8888'
        url = APIHOST + '/api/v1/namespaces/' + NAMESPACE + '/actions/' + ACTION

        response = requests.post(url, json=PARAMS, params={'blocking': BLOCKING, 'result': RESULT}, auth=(user_pass[0], user_pass[1]))
        return response.text

def main(args):
    global nninvoker
    nninvoker = NNInvoker(args)
    if(complete):
        testre = nninvoker.invokeTest()
        return {'status': testre}
    else:
        testre = nninvoker.trainNN()
        return {"activations": testre}
