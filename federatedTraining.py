import random
import pickle
import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


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


def trainFederated(X_train,y_train,comms_round,global_model,local_model,clients_data):
    # commence global training loop
    client_names = clients_data.keys()
    local_models = []

    for comm_round in range(comms_round):
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.coef_
        global_b = global_model.intercept_
        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        scaled_local_b_list = list()

        # randomize client data
        random.shuffle(list(client_names))
        # loop through each client and create new local model
        for client in client_names:
            # set local model weight to the weight of the global model
            client_data = clients_data[client]
            X_train , y_train = list(map(list, zip(*client_data)))

            # fit local model with client's data
            local_model.fit(X_train,y_train,coef_init=global_weights,intercept_init=global_b)

            # scale the model weights and add to list
            scaling_factor = 1/len(client_names)
            scaled_weights = scale_model_weights(local_model.coef_, scaling_factor)
            scaled_b = scale_model_weights(local_model.intercept_, scaling_factor)

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

        # test global model and print out metrics after each communications round
        #for (X_test, Y_test) in test_batched:
        #    global_acc, global_loss = test_model(X_test, Y_test, global_model, comm_round)
        test_pred = global_model.predict(X_train)
        print(accuracy_score(y_train, test_pred))
