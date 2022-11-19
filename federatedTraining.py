import random
import numpy as np
from sklearn.metrics import accuracy_score
import copy


'''
from sklearn.svm import SVC
clf = SVC(C = 1e5, kernel = 'linear')
clf.fit(X, y) 
print('w = ',clf.coef_)
print('b = ',clf.intercept_)
print('Indices of support vectors = ', clf.support_)
print('Support vectors = ', clf.support_vectors_)
print('Number of support vectors for each class = ', clf.n_support_)
print('Coefficients of the support vector in the decision function = ', np.abs(clf.dual_coef_))
'''

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
    # get the average grad accross all client gradients
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




def trainFederated(X_test,y_test,comms_round,global_model,local_models,clients_data):
    # commence global training loop
    client_names = clients_data.keys()
    for comm_round in range(comms_round):
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = copy.deepcopy(global_model.coef_)
        global_b = copy.deepcopy(global_model.intercept_)
        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        scaled_local_b_list = list()

        # randomize client data
        random.shuffle(list(client_names))
        # loop through each client and create new local model
        print(comm_round)
        client_num = len(client_names)
        client_names = list(client_names)
        for id in range(client_num):
            client = client_names[id]
            client_data = clients_data[client]
            X_train, y_train = list(map(list, zip(*client_data)))
            # set local model weight to the weight of the global model and fit local model with client's data
            local_models[id].fit(X_train,y_train,coef_init=copy.deepcopy(global_weights),intercept_init=copy.deepcopy(global_b))

            test_pred = local_models[id].predict(X_test)
            print(accuracy_score(y_test, test_pred))

            # scale the model weights and add to list
            scaling_factor = 1/len(client_names)
            scaled_weights = scale_model_weights(local_models[id].coef_, scaling_factor)
            scaled_b = scale_model_weights(local_models[id].intercept_, scaling_factor)

            scaled_local_weight_list.append(scaled_weights)
            scaled_local_b_list.append(scaled_b)

            # clear session to free memory after each communication round
            #K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_local_weight_list)
        average_b = sum_scaled_weights(scaled_local_b_list)

        # update global model
        global_model.coef_ = average_weights
        global_model.intercept_ = average_b

    return global_model,local_models
