from ..utils.client import create_clients
from ..utils.readFile import readLabel,readData,readCounts2CPM,removeZerogene
from federatedACTINN import trainFederated,actinnmodel,trainLocal
from sklearn.model_selection import train_test_split
from ..utils.intraPara import get_Datasets
#from eval import testSet
from ..utils.writeFile import outResults,outSta
from ..utils.readFile import selRepsample
import numpy as np
from eval_actinn import testSet
import time


dir = '../Intra-dataset/'
resultdir = '../results/'

datasetid ={1:'AMB',2:'BaronHuman',3:'BaronMouse',4:'Muraro',5:'Segerstolpe',6:'Xin',7:'TM',8:'zhengsorted'}


def runactinnmain(num_clients,dataid,randomint,runlabel):
    print('here')
    dataset = datasetid[int(dataid)]
    datasets = get_Datasets()
    datafile = dir + datasets[dataset][0]
    labelfile = dir + datasets[dataset][1]

    X,cells,colnames = readData(datafile)
    y = readLabel(labelfile)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=randomint)
    X_samples, y_samples = selRepsample(X_train,y_train)

    clients = create_clients(X_samples, y_samples,X_train, y_train, num_clients=num_clients, initial='client')
    print(clients.keys())

    #initializatioin global and local parameters
    global_parameters = None

    local_parameters = []
    for i in range(num_clients):
        local_parameters.append(None)
    # fit global, local and center models
    global_parameters, global_time1 = trainFederated(
        X_test=X_test,
        y_test=y_test,
        comms_round=comms_round,
        global_parameters=global_parameters,
        local_parameters=local_parameters,
        clients_data=clients)


    train_set = np.transpose(X_train)
    test_set = np.transpose(X_test)

    center_start_time = time.time()
    center_parameters,label_to_type_dict = actinnmodel(train_set,y_train,parameters=None,epoch=comms_round)
    center_end_time = time.time()
    center_time = center_end_time - center_start_time


    local_start_time = time.time()
    local_parameters = trainLocal(local_parameters,clients_data=clients,epoch=comms_round)
    local_end_time = time.time()
    local_time = local_end_time - local_start_time

    times = [center_time, local_time, global_time1]
    #output results

    results = testSet(global_parameters,local_parameters,center_parameters,label_to_type_dict,test_set,y_test)

    outResults(resultdir,dataset,modelname,comms_round,num_clients,results,runlabel,randomint,times)

    outSta(y,y_train,y_test,clients,resultdir, dataset,modelname,comms_round,num_clients,randomint)


if __name__ == '__main__':
    modelname = 'actinn'
    testratio = 0.2
    comms_round = 100

    runlabel = 'test0928'
    #for  randomint in [10,20,30,42,50]:
    #    for num_clients in [2,5,10,20]:
    #        runactinnmain(num_clients, dataid, randomint, runlabel)

    for dataid in [4]:
        for num_clients in [2]:
            for randomint in [10]:
                runactinnmain(num_clients, dataid, randomint, runlabel)