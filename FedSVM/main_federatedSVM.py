from ..utils.client import create_clients
from ..utils.readFile import readLabel,readData,readCounts2CPM,removeZerogene
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from ..utils.intraPara import get_Datasets
from ..utils.centerModel import centerSVM
from ..utils.eval import testSet
from ..utils.writeFile import outResults,outSta
from ..utils.readFile import selRepsample
import time
from federatedSVM import trainFederated,trainLocal
import pickle
from sklearn.metrics import accuracy_score


dir = '../Intra-dataset/'
resultdir = '../results/'

datasetid ={1:'AMB',2:'BaronHuman',3:'BaronMouse',4:'Muraro',5:'Segerstolpe',6:'Xin',7:'TM',8:'zhengsorted'}


def runsvmmain(num_clients,dataid,randomint,runlabel,runtime=False):

    dataset = datasetid[int(dataid)]
    print(dataset)
    global_model_file  = modeldir + 'global' + modelname + '_' + dataset + '_' + str(num_clients) + '_' +runlabel + '_' + str(randomint)
    datasets = get_Datasets()

    datafile = dir + datasets[dataset][0]
    labelfile = dir + datasets[dataset][1]

    X, cells, colnames = readData(datafile)
    y = readLabel(labelfile)
    import collections
    print(str(len(y))+' '+str(len(collections.Counter(y))))


    X = removeZerogene(X)
    X = readCounts2CPM(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=randomint)
    X_samples, y_samples = selRepsample(X_train, y_train)

    clients = create_clients(X_samples, y_samples, X_train, y_train, num_clients=num_clients, initial='client')
    print(clients.keys())

    # initializatioin global and local models
    global_clf = SGDClassifier(random_state=0,learning_rate = 'constant', eta0 = 0.000001,alpha = 0.5,
                               class_weight = 'balanced',loss='hinge')

    global_clf.fit(X_samples, y_samples)
    with open(global_model_file, 'wb') as f:
        pickle.dump(global_clf, f)

    # fit global, local and center models
    global_start_time = time.time()
    global_model, global_time1 = trainFederated(X_test,y_test,comms_round, global_model_file=global_model_file, clients_data=clients)
    test_pred = global_model.predict(X_test)
    print(accuracy_score(y_test, test_pred))

    global_end_time = time.time()
    global_time = global_end_time - global_start_time

    print(global_time1)
    print(global_time)

    local_start_time = time.time()
    local_models =trainLocal(clients,comms_round)
    local_end_time = time.time()
    local_time = local_end_time - local_start_time

    center_start_time = time.time()
    center_model = centerSVM(comms_round, X_train, y_train)
    center_end_time = time.time()
    center_time = center_end_time - center_start_time

    times = [center_time, local_time, global_time1]

    # output results
    results = testSet(global_model, local_models, center_model, X_test, y_test)

    if runtime == False:
        times = []
    outResults(resultdir, dataset, modelname, comms_round, num_clients, results, runlabel, randomint,times)

    outSta(y, y_train, y_test, clients, resultdir, dataset, modelname, comms_round, num_clients ,randomint)



if __name__ == '__main__':
    modelname = 'svm'
    testratio = 0.2
    modeldir = './models/'
    comms_round = 100
    randomint = 10
    #num_clients = 5
    #dataid = 4
    runlabel = 'test0928'
    for dataid in [1,2,3,4,5,6,7,8]:
        dataid = 4
        num_clients = 2
        randomint = 10
        for num_clients in [2, 5, 10, 20]:
            for randomint in [10,20,30,42,50]:# five time run
                runsvmmain(num_clients, dataid, randomint, runlabel, runtime=True)