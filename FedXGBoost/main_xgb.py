#main file
from client import create_clients
from readFile import readLabel,readData,readCounts2CPM,removeZerogene
from federatedTraining_xgb import trainFederated
#from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from intraPara import get_Datasets
#from centerModel import centerSVM
from eval_xgb import testSet
from writeFile import outResults,outSta
from readFile import selRepsample
from sklearn import preprocessing
from federated_client import localmodel,centermodel
import time
#import multiprocessing
#from os import kill
from os import getpid
#from signal import SIGKILL

#dir = '../sigGCN-main/scRNAseq_Benchmark_datasets/Intra-dataset/'
#resultdir ='../sigGCN-main/scRNAseq_Benchmark_datasets/results/'

dir = '../Intra-dataset/'
resultdir ='../results/'

#dir = '/scratch/qsu226/FL/Intra-dataset/'
#resultdir = '/scratch/qsu226/FL/results/'

datasetid ={1:'AMB',2:'BaronHuman',3:'BaronMouse',4:'Muraro',5:'Segerstolpe',6:'Xin',7:'TM',8:'zhengsorted'}

modelname = 'xgb'
comms_round = 100
testratio = 0.2



def runxgbmain(num_clients,dataid,randomint,runlabel,runtime=False):
    print('in xgb main')
    modeldir = '../models/'

    dataset = datasetid[int(dataid)]
    datasets = get_Datasets()

    port = dataid + randomint + 9000 + num_clients*100

    globalmodelname = modeldir + 'global' + modelname + '_' + dataset + '_' + str(num_clients) + '_' +runlabel + '_' + str(randomint)
    localmodelname = modeldir + 'local' + modelname + '_' + dataset + '_' + str(num_clients) + '_' +runlabel + '_' + str(randomint)
    centermodelname = modeldir + 'center' + modelname + '_' + dataset + '_' + str(num_clients) + '_' +runlabel + '_' + str(randomint)

    datafile = dir + datasets[dataset][0]
    labelfile = dir + datasets[dataset][1]

    X,cells,colnames = readData(datafile)
    y = readLabel(labelfile)
    X = removeZerogene(X)
    X = readCounts2CPM(X)

    le = preprocessing.LabelEncoder()
    le.fit(list(set(y)))
    y = le.transform(y)
    class_num = len(set(y))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=randomint)
    X_samples, y_samples = selRepsample(X_train,y_train)

    clients = create_clients(X_samples, y_samples,X_train, y_train, num_clients=num_clients, initial='client')
    print(clients.keys())

    print(globalmodelname+'\n\n')
    print('port: '+str(port)+'\n')

    global_start_time = time.time()
    print('Global model training')
    trainFederated(X_test,y_test,clients,class_num,globalmodelname,port)
    global_end_time = time.time()
    global_time = global_end_time - global_start_time

    local_start_time = time.time()
    print('Local model training')
    localmodel(localmodelname, X_test, y_test, clients, class_num)
    local_end_time = time.time()
    local_time = local_end_time - local_start_time

    center_start_time = time.time()
    print('Centered model training')
    centermodel(centermodelname, X_train, y_train, X_test, y_test, class_num)
    center_end_time = time.time()
    center_time = center_end_time - center_start_time

    times = [center_time, local_time, global_time]

    if runtime == False:
        times = []

    #output results

    print(center_time)
    print(local_time)
    print(global_time)

    results = testSet(clients,globalmodelname, localmodelname, centermodelname,X_test,y_test)

    outResults(resultdir,dataset,modelname,comms_round,num_clients,results,runlabel,randomint, times)

    outSta(y,y_train,y_test,clients,resultdir, dataset,modelname,comms_round,num_clients,randomint)

    #print('Finish. Kill process\n')
    # get the pid of the current process
    #pid = getpid()
    # report a message
    #print(f'Running with pid: {pid}')
    # attempt to kill the current process
    #kill(pid, SIGKILL)
    # report a message
    #print('Skill running')

num_clients = 10
dataset = 3
randomint = 10
runlabel = '0404'
runxgbmain(num_clients, dataset, randomint, runlabel,True)


