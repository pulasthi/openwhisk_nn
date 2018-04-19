import couchdb

user = "whisk_admin"
password = "some_passw0rd"
couchserver = couchdb.Server("http://%s:%s@127.0.0.1:5984/" % (user, password))
dbname= 'digitnndb'
db = couchserver[dbname]

resultsDoc = db.get('finalResults')
iterDoc = db.get('iterResults')
print iterDoc
stimedoc = db.get('startTime')
etimedoc = db.get('endTime')
stime = stimedoc['stime']
etime = etimedoc['etime']
totalTime = etime - stime
print('Time taken : ' + str(totalTime))
print(resultsDoc['result'])
