#!/usr/bin/python
import multiprocessing
import sys
import time

import xgboost as xgb
import xgboost.federated

SERVER_KEY = 'server-key.pem'
SERVER_CERT = 'server-cert.pem'
CLIENT_KEY = 'client-key.pem'
CLIENT_CERT = 'client-cert.pem'

#federated client
#46

NUM_ROUND = 30
early_stopping_rounds = None
def localmodel(modelname,X_test,y_test,clients_data,class_num):
    client_names = list(clients_data.keys())

    for rank in range(len(client_names)):
        localmodelname = modelname + '_' +str(rank)
        client = client_names[rank]
        client_data = clients_data[client]
        X_train, y_train = list(map(list, zip(*client_data)))

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)

        # Specify parameters via map, definition are same as c++ version
        param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softprob', 'num_class': class_num}

        # Specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = NUM_ROUND

        # Run training, all the features in training API is available.
        bst = xgb.train(param, dtrain, num_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds)
        bst.save_model(localmodelname)


def centermodel(modelname, X_train,y_train,X_test,y_test,class_num):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Specify parameters via map, definition are same as c++ version
    param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softprob', 'num_class': class_num}

    # Specify validations set to watch performance
    watchlist = [(dtest, 'eval'), (dtrain, 'train')]
    num_round = NUM_ROUND

    # Run training, all the features in training API is available.
    bst = xgb.train(param, dtrain, num_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds)
    bst.save_model(modelname)




def run_worker(port: int, world_size: int, rank: int, with_ssl: bool, with_gpu: bool, clients_data: dict, class_num : int, testdata: list, modelname: str) -> None:
    print('client' + str(rank))
    communicator_env = {
        'xgboost_communicator': 'federated',
        'federated_server_address': f'federated_server.com:{port}',
        'federated_world_size': world_size,
        'federated_rank': rank
    }
    if with_ssl:
        communicator_env['federated_server_cert'] = SERVER_CERT
        communicator_env['federated_client_key'] = CLIENT_KEY
        communicator_env['federated_client_cert'] = CLIENT_CERT

    # Always call this before using distributed module
    with xgb.collective.CommunicatorContext(**communicator_env):
        # Load file, file will not be sharded in federated mode.
        client_names = list(clients_data.keys())
        client = client_names[rank]
        client_data = clients_data[client]
        X_train, y_train = list(map(list, zip(*client_data)))

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(testdata[0],label=testdata[1])

        # Specify parameters via map, definition are same as c++ version
        param = {'max_depth': 2, 'eta': 1, 'objective': 'multi:softprob','num_class': class_num}
        if with_gpu:
            param['tree_method'] = 'gpu_hist'
            param['gpu_id'] = rank

        # Specify validations set to watch performance
        watchlist = [(dtest, 'eval'), (dtrain, 'train')]
        num_round = NUM_ROUND

        # Run training, all the features in training API is available.
        bst = xgb.train(param, dtrain, num_round, evals=watchlist,early_stopping_rounds=early_stopping_rounds)
        #
        # Save the model, only ask process 0 to save the model.
        if xgb.collective.get_rank() == 0:
            bst.save_model(modelname)
            xgb.collective.communicator_print("Finished training\n")



if __name__ == '__main__':
    world_size = int(sys.argv[1])
    rank = int(sys.argv[2])
    port = int(sys.argv[3])
    #run_worker(port, world_size, rank, with_ssl=False, with_gpu=False,clients_data)
