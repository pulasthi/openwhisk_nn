from subprocess import check_output
import sys
import json
import couchdb

user = "whisk_admin"
password = "some_passw0rd"
couchserver = couchdb.Server("http://%s:%s@149.165.150.85:30301/" % (user, password))
dbname= 'digitnndb'
db = couchserver[dbname]

resultsDoc = db.get('finalResults')
stimedoc = db.get('startTime')
etimedoc = db.get('endTime')
stime = stimedoc['stime']
etime = etimedoc['etime']
totalTime = etime - stime
print('Time taken : ' + str(totalTime))
accustr = resultsDoc['result']
sindex = accustr.find('lts:')
eindex = accustr.find('/')
accuracy = accustr[sindex+5:eindex]

#call("wsk -i activation get 75d69f6ee4ae4ec0969f6ee4ae5ec0be", shell=True])
callval = check_output(["wsk", "-i","activation","get",sys.argv[1]])
sindex = callval.find("{")
eindex = callval.rfind("}")
callval = callval[sindex:eindex+1]
jsonval = json.loads(callval)

print 'Driver time {0}'.format(jsonval['duration'])
traineractIds = jsonval['response']['result']['activations']
count = 0
totalfunctions = len(traineractIds)
funcTotalSum = 0
funcTotalMax = 0
funcDataSum = 0
funcTrainSum = 0
starttimes = []

for activation in traineractIds:
    sin = activation.find('":"')
    ein = activation.rfind('"')
    actId = activation[sin+3:ein]
    subcallval = check_output(["wsk", "-i","activation","get",actId])
    sindex = subcallval.find("{")
    eindex = subcallval.rfind("}")
    subcallval = subcallval[sindex:eindex+1]
    subjsonval = json.loads(subcallval)
    logs = subjsonval['logs']

    #time2 dataload
    t2start = logs[1].find("time2 - ") + 8
    time2str = logs[1][t2start:]

    #time5 dataload
    t5start = logs[len(logs) - 1].find("time5 - ") + 8
    time5str = logs[len(logs) - 1][t5start:]
    funcTotalMax = max(funcTotalMax, int(subjsonval['duration']))
    funcTotalSum += int(subjsonval['duration'])
    funcDataSum += float(time2str)
    funcTrainSum += float(time5str)
    starttimes.append(long(subjsonval['start'])/1000)
    #print 'Trainer {0} total time {1}'.format(count,subjsonval['duration'])
    #print 'Trainer {0} start time(s) {1}'.format(count,str(long(subjsonval['start'])/1000))
    #print 'Trainer {0} end time(s) {1}'.format(count,str(long(subjsonval['end'])/1000))
    #print 'Trainer {0} time2 {1}'.format(count,str(float(time2str)))
    #print 'Trainer {0} time5 {1}'.format(count,str(float(time5str)))
    count += 1

minstart = min(starttimes)
starttimes[:] = [x - minstart for x in starttimes]
print 'start time variations {0}'.format(starttimes)

print '{0}\t{1}\t{2}\t{3}\t{4}\t{5}'.format(totalTime,accuracy,funcTotalSum/totalfunctions,funcDataSum/totalfunctions,funcTrainSum/totalfunctions,funcTotalMax)
