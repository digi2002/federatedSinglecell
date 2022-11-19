import random
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from actinn import scale_sets,type_to_label_dict1,convert_type_to_label,one_hot_matrix,model,predict
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final

def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad across all client gradients
    for weight_list in zip(*scaled_weight_list):
        weight_array = np.array(weight_list)
        weight_mean = np.sum(weight_array, axis=0)
        avg_grad.append(weight_mean)
    return np.array(avg_grad)

def testModel(global_model,local_models,X_test,y_test):
    local_accs = []
    global_pred = global_model.predict(X_test)
    global_acc = accuracy_score(global_pred, y_test)
    for local_model in local_models:
        local_pred = local_model.predict(X_test)
        local_acc = accuracy_score(local_pred, y_test)
        print(local_acc)
        local_accs.append(local_acc)
    print(global_acc)
    return global_acc, np.average(np.array(local_accs))


def actinnmodel(train_set,train_label,parameters,epoch):
    nt = len(set(train_label))
    #train_set, test_set = scale_sets([train_set, test_set])
    type_to_label_dict = type_to_label_dict1(train_label)
    label_to_type_dict = {v: k for k, v in type_to_label_dict.items()}
    print("Cell Types in training set:", type_to_label_dict)
    print("# Cells in training set:", len(train_label))
    print("# Genes in training set:", str(train_set.shape[0]))
    train_label = convert_type_to_label(train_label, type_to_label_dict)
    train_label = one_hot_matrix(train_label, nt)
    parameters = model(train_set, train_label,parameters=parameters,num_epochs = epoch)
    return parameters,label_to_type_dict

def trainFederated(comms_round,global_parameters,local_parameters,clients_data,epoch=1):
    # commence global training loop
    client_names = clients_data.keys()
    for comm_round in range(comms_round):
        # get the global model's weights - will serve as the initial weights for all local models
        parameters_list = {"W1": [],
                           "b1": [],
                           "W2": [],
                           "b2": [],
                           "W3": [],
                           "b3": [],
                           "W4": [],
                           "b4": []}

        # randomize client data
        random.shuffle(list(client_names))
        # loop through each client and create new local model
        print(comm_round)
        client_num = len(client_names)
        client_names = list(client_names)
        for id in range(client_num):
            print('client id' + str(id))
            client = client_names[id]
            client_data = clients_data[client]
            X_train, y_train = list(map(list, zip(*client_data)))

            train_set = np.transpose(X_train)
            #test_set = np.transpose(X_test)
            #train_set = pd.DataFrame(np.transpose(X_train), index=colnames)
            #test_set = pd.DataFrame(np.transpose(X_test), index=colnames)

            # set local model weight to the weight of the global model and fit local model with client's data
            local_parameters[id],type_to_label_dict = actinnmodel(train_set, y_train, global_parameters,epoch=epoch)

            # scale the model weights and add to list
            scaling_factor = 1/len(client_names)
            for para_key in local_parameters[id].keys():
                scaled_para = scale_model_weights(local_parameters[id][para_key],scaling_factor)
                parameters_list[para_key].append(scaled_para)

            # clear session to free memory after each communication round
            #K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        global_parameters = {"W1": None,
                           "b1": None,
                           "W2": None,
                           "b2": None,
                           "W3": None,
                           "b3": None,
                           "W4": None,
                           "b4": None}

        for para_key in parameters_list.keys():
            global_parameters[para_key] = sum_scaled_weights(parameters_list[para_key])

    return global_parameters




def trainLocal(local_parameters,clients_data,epoch):
    # commence global training loop
    client_names = clients_data.keys()

    # randomize client data
    random.shuffle(list(client_names))
    # loop through each client and create new local model
    client_num = len(client_names)
    client_names = list(client_names)
    for id in range(client_num):
        print('client id' + str(id))
        client = client_names[id]
        client_data = clients_data[client]
        X_train, y_train = list(map(list, zip(*client_data)))
        train_set = np.transpose(X_train)

        local_parameters[id], type_to_label_dict = actinnmodel(train_set, y_train, None,epoch=epoch)

    return local_parameters



def actininfer(test_set,test_label,parameters,label_to_type_dict):
    test_predict = predict(test_set, parameters)
    predicts = []
    for i in range(len(test_predict)):
        predicts.append(label_to_type_dict[test_predict[i]])
    truths = test_label
    acc=accuracy_score(truths, predicts)
    f1 = f1_score(truths, predicts, average='weighted')
    recall = recall_score(truths, predicts, average='weighted')
    precision = precision_score(truths, predicts, average='weighted')
    outstr=str(acc)+','+str(f1)+','+str(recall)+','+str(precision)+'\n'
    print(outstr)





