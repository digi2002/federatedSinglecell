from client import create_clients
from readFile import readLabel,readData,readCounts2CPM,removeZerogene
from federatedTraining_xgb import trainFederated
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from intraPara import get_Datasets
from centerModel import centerSVM
from eval_xgb import testSet
from writeFile import outResults,outSta
import sys
from readFile import selRepsample
from sklearn import preprocessing
from federated_client import localmodel,centermodel

#dir = '../sigGCN-main/scRNAseq_Benchmark_datasets/Intra-dataset/'
#resultdir ='../sigGCN-main/scRNAseq_Benchmark_datasets/results/'

dir = '../Intra-dataset/'
resultdir ='../results/'

#dir = '/scratch/qsu226/FL/Intra-dataset/'
#resultdir = '/scratch/qsu226/FL/results/'

datasetid ={1:'AMB',2:'BaronHuman',3:'BaronMouse',4:'Muraro',5:'Segerstolpe',6:'Xin',7:'TM',8:'zhengsorted'}

modelname = 'xgb'
comms_round = 0
testratio = 0.2

num_clients = 2
dataset = 'Xin'

globalmodelname = 'global' + modelname + '_' + dataset + '_' + str(num_clients)
localmodelname = 'local' + modelname + '_' + dataset + '_' + str(num_clients)
centermodelname = 'center' + modelname + '_' + dataset + '_' + str(num_clients)

#dataset = datasetid[int(sys.argv[1])]
#num_clients = int(sys.argv[2])
datasets = get_Datasets()

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=42)
X_samples, y_samples = selRepsample(X_train,y_train)

clients = create_clients(X_samples, y_samples,X_train, y_train, num_clients=num_clients, initial='client')
print(clients.keys())

'''
print('Local model training')
localmodel(localmodelname,X_test,y_test,clients,class_num)

print('Centered model training')
centermodel(centermodelname, X_train,y_train,X_test,y_test,class_num)
'''
print('Global model training')
trainFederated(X_test,y_test,clients,class_num,globalmodelname)


#output results
results = testSet(clients,globalmodelname, localmodelname, centermodelname,X_test,y_test)

outResults(resultdir,dataset,modelname,comms_round,num_clients,results,'1119')

outSta(y,y_train,y_test,clients,resultdir, dataset,modelname,comms_round,num_clients)




'''


import numpy as np
from sklearn.metrics import precision_score, recall_score, accuracy_score



results = testSet(global_model,local_models,center_model,X_test,y_test)


preds = model.predict(D_test)
best_preds = np.asarray([np.argmax(line) for line in preds])

print("Precision = {}".format(precision_score(y_test, best_preds, average='macro')))
print("Recall = {}".format(recall_score(y_test, best_preds, average='macro')))
print("Accuracy = {}".format(accuracy_score(y_test, best_preds)))





#initializatioin global and local models
global_clf = SGDClassifier(random_state=0,loss='hinge',max_iter=1)
global_clf.fit(X_train,y_train)

local_clfs = []
for i in range(num_clients):
    local_clfs.append(SGDClassifier(random_state=0,loss='hinge',max_iter=1,warm_start=True))
# fit global, local and center models
global_model,local_models = trainFederated(X_test,y_test,comms_round,global_model=global_clf,local_models=local_clfs,clients_data=clients)
center_model = centerSVM(comms_round,X_train, y_train)

#output results
results = testSet(global_model,local_models,center_model,X_test,y_test)

outResults(resultdir,dataset,modelname,comms_round,num_clients,results,'920')

outSta(y,y_train,y_test,clients,resultdir, dataset,modelname,comms_round,num_clients)
'''