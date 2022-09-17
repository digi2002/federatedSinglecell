from client import create_clients
from readFile import readTrain,readCounts2CPM,removeZerogene
from federatedTraining import trainFederated
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split


trainfile = './singlecell/train.csv'

comms_round = 20
testratio = 0.2
num_clients = 5

X,y = readTrain(trainfile)
X = removeZerogene(X)
X = readCounts2CPM(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testratio, random_state=42)


clients = create_clients(X_train, y_train, num_clients=num_clients, initial='client')
print(clients.keys())

global_clf = SGDClassifier(random_state=0,loss='hinge',max_iter=1)
global_clf.fit(X_train,y_train)
local_clf = SGDClassifier(random_state=0,loss='hinge',max_iter=1, warm_start=True)

trainFederated(X_test,y_test,comms_round,global_model=global_clf,local_model=local_clf,clients_data=clients)