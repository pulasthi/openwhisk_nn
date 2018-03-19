import cPickle
import gzip
import couchdb
import json
import numpy as np

"""
Uploads the datainto the couchdb database. Also parititions the dataset into a number of smaller datasets
"""
para = 20
f = gzip.open('mnist.pkl.gz', 'rb')
training_data, validation_data, test_data = cPickle.load(f)
f.close()


print(len(training_data))
print(len(training_data[0]))
print(len(training_data[1]))
print(validation_data[0])
user = "whisk_admin"
password = "some_passw0rd"
couchserver = couchdb.Server("http://%s:%s@127.0.0.1:5984/" % (user, password))
dbname= 'digitnndata'

if dbname in couchserver:
    del couchserver[dbname]

db = couchserver.create(dbname)

testdataSize = 5000;
# make val and test data savable
testdata_1 = test_data[0].tolist()
testdata_2 = test_data[1].tolist()
testtuple = (testdata_1[0:testdataSize], testdata_2[0:testdataSize])
print('testset len')
print(len(testtuple[0]))

validata_1 = validation_data[0].tolist()
validata_2 = validation_data[1].tolist()
valdatatuple = (validata_1, validata_2)

# break down trainning dataset
traindDataPart = []
totalLen = len(training_data[0])
partLen = totalLen/para
traindatalist1 = training_data[0].tolist()
traindatalist2 = training_data[1].tolist()

for x in range(para):
    startin = x*partLen
    endin = startin + partLen
    temp1 = traindatalist1[startin:endin]
    temp2 = traindatalist2[startin:endin]
    traindDataPart.append((temp1, temp2))

print(len(traindDataPart))
print(len(traindDataPart[0]))
print(len(traindDataPart[0][1]))

validationData = 'valdata'
testData = 'testdata'

#only upload first 4 for now
for x in range(4):
    trainData = 'traindata' + str(x)
    traindoc = {'data': traindDataPart[x]}
    db[trainData] = traindoc

#valdoc = {'data': valdatatuple}
testdoc = {'data': testtuple}
print('done3')

#db[validationData] = valdoc
print('done1')

db[testData] = testdoc
print('done2')
