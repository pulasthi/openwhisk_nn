import cPickle
import gzip
import couchdb
import json
import numpy as np

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

user = "whisk_admin"
password = "some_passw0rd"
couchserver = couchdb.Server("http://%s:%s@149.165.150.85:30301/" % (user, password))
dbname= 'initdatadb'

if dbname in couchserver:
    del couchserver[dbname]

db = couchserver.create(dbname)

#better method for weight initb
sizes = [784, 16, 16, 10]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x)/np.sqrt(x)
                for x, y in zip(sizes[:-1], sizes[1:])]
# save the initialized variables in couchdb

wdoc = {'w': convertToJSON(weights)}
bdoc = {'b': convertToJSON(biases)}

db['constw'] = wdoc
db['constb'] = bdoc
