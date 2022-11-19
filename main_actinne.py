from client import create_clients
from readFile import readLabel,readData,readCounts2CPM,removeZerogene
from federatedTraining import trainFederated
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from intraPara import get_Datasets
from centerModel import centerSVM
from eval import testSet
from writeFile import outResults,outSta
import sys
from readFile import selRepsample
import numpy as np
import pandas as pd
from federatedTraining_actinn import trainFederated,actinnmodel,trainLocal
from eval_actinn import testSet

#dir = '../sigGCN-main/scRNAseq_Benchmark_datasets/Intra-dataset/'
#resultdir ='../sigGCN-main/scRNAseq_Benchmark_datasets/results/'
dir = '/scratch/qsu226/FL/Intra-dataset/'
resultdir = '/scratch/qsu226/FL/results/'

datasetid ={1:'AMB',2:'BaronHuman',3:'BaronMouse',4:'Muraro',5:'Segerstolpe',6:'Xin',7:'TM',8:'zhengsorted'}

modelname = 'actinn'
comms_round = 100
testratio = 0.2

#num_clients = 2
#dataset = 'Xin'

dataset = datasetid[int(sys.argv[1])]
num_clients = int(sys.argv[2])

datasets = get_Datasets()

datafile = dir + datasets[dataset][0]
labelfile = dir + datasets[dataset][1]

X,cells,colnames = readData(datafile)
y = readLabel(labelfile)
#X = removeZerogene(X)
#X = readCounts2CPM(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=42)
X_samples, y_samples = selRepsample(X_train,y_train)

clients = create_clients(X_samples, y_samples,X_train, y_train, num_clients=num_clients, initial='client')
print(clients.keys())

#initializatioin global and local parameters
global_parameters = None
local_parameters = []
for i in range(num_clients):
    local_parameters.append(None)
# fit global, local and center models
global_parameters = trainFederated(
               comms_round=comms_round,
               global_parameters = global_parameters ,
               local_parameters = local_parameters,
               clients_data=clients)

train_set = np.transpose(X_train)
test_set = np.transpose(X_test)

center_parameters,label_to_type_dict = actinnmodel(train_set,y_train,parameters=None,epoch=comms_round)

local_parameters = trainLocal(local_parameters,clients_data=clients,epoch=comms_round)

#output results
results = testSet(global_parameters,local_parameters,center_parameters,label_to_type_dict,test_set,y_test)

outResults(resultdir,dataset,modelname,comms_round,num_clients,results,'102')

outSta(y,y_train,y_test,clients,resultdir, dataset,modelname,comms_round,num_clients)

