#save

import random
import numpy as np
from sklearn.metrics import accuracy_score
import copy

from ..utils.centerModel import centerSVM

import multiprocessing
import time
import socket
import pickle
from sklearn.linear_model import SGDClassifier


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


def receiv(sock):
    # Receive the response header and data
    response_header = sock.recv(4)
    response_length = int.from_bytes(response_header, byteorder='big')
    response_data = b''

    while len(response_data) < response_length:
        data = sock.recv(response_length - len(response_data))
        if not data:
            break
        response_data += data
    return response_data

def send(sock,data):
    # Create a message header with the length of the message data
    header = len(data).to_bytes(4, byteorder='big')

    # Send the message header and data
    sock.sendall(header + data)


def send_model(sock,model):
    model_data = pickle.dumps(model)

    # Send the model
    header = len(model_data).to_bytes(4, byteorder='big')
    sock.sendall(header + model_data)

def receive_model(sock):
    model_header = sock.recv(4)
    model_length = int.from_bytes(model_header, byteorder='big')
    model_data = b''
    while len(model_data) < model_length:
        data = sock.recv(model_length - len(model_data))
        if not data:
            break
        model_data += data

    # Deserialize the model
    model = pickle.loads(model_data)
    return model


def run_server(X_test,y_test,HOST,PORT,comms_round,global_model_file,clients_data,shared_dict):
    global_start_time = time.time()
    client_num = len(clients_data.keys())
    with open(global_model_file, 'rb') as f:
        global_model = pickle.load(f)
    # global_model,Initialize the global model
    # Listen for incoming connections from clients
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(client_num*comms_round)
        print('Server started and listening...')
        client_sockets = []
        while len(client_sockets) < client_num:
            # accept a client connection
            client_socket, addr = server_socket.accept()
            print("Got connection from", addr)
            # add the client socket to the list
            client_sockets.append(client_socket)
            send_model(client_socket,global_model)
            print('intial gobal model have been sent to client ')

        # Start the training loop
        for round in range(comms_round):
            scaled_local_weight_list = []
            scaled_local_b_list = []
            for client_id in range(client_num):
                client_socket = client_sockets[client_id]
                # Receive the updated model weights from client i
                local_model = receive_model(client_socket)
                print('round '+str(round)+' server received scaled local model from client '+str(client_id))

                scaled_local_weight_list.append(local_model.coef_)
                scaled_local_b_list.append(local_model.intercept_)

            # to get the average over all the local model, we simply take the sum of the scaled weights
            average_weights = sum_scaled_weights(scaled_local_weight_list)
            average_b = sum_scaled_weights(scaled_local_b_list)

            global_model.coef_ = copy.deepcopy(average_weights)
            global_model.intercept_ = copy.deepcopy(average_b)

            for client_id in range(client_num):
                client_socket = client_sockets[client_id]
                send_model(client_socket,global_model)
                print('round '+str(round)+' server send updated global model to client '+str(client_id))

            test_pred = global_model.predict(X_test)
            print('round ' + str(round) + ' global performance')
            print(accuracy_score(y_test, test_pred))

            shared_dict['result'] = global_model
            global_end_time = time.time()
            global_time = global_end_time - global_start_time
            shared_dict['time'] = global_time

        #test_pred = global_model.predict(X_test)
        #print('global performance in final')
        #print(accuracy_score(y_test, test_pred))

        #with open(global_model_file+'trained', 'wb') as f:
        #    pickle.dump(global_model, f)

        server_socket.close()



def run_client(HOST,PORT,comms_round,clients_data, client_names, client_id,count):
    local_model = SGDClassifier(random_state=0, loss='hinge',max_iter=1,
                                        learning_rate = 'constant', eta0 = 0.000001, alpha = 0.5,
                                        class_weight = 'balanced',warm_start=True)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        # Receive the initial global model from the server
        global_model = receive_model(client_socket)
        global_weights = copy.deepcopy(global_model.coef_)
        global_b = copy.deepcopy(global_model.intercept_)

        for i in range(comms_round):
            # Train the local model
            client = client_names[client_id]
            client_data = clients_data[client]
            X_train, y_train = list(map(list, zip(*client_data)))
            local_model.fit(X_train, y_train, coef_init=copy.deepcopy(global_weights),
                                 intercept_init=copy.deepcopy(global_b))

            if count != 0:
                scaling_factor = len(y_train) / count
            else:
                scaling_factor = 1 / len(client_names)
            scaled_weights = scale_model_weights(local_model.coef_, scaling_factor)
            scaled_b = scale_model_weights(local_model.intercept_, scaling_factor)

            # Send the updated model to the server
            local_model.coef_ = copy.deepcopy(scaled_weights)
            local_model.intercept_ = copy.deepcopy(scaled_b)
            send_model(client_socket,local_model)
            print('round '+ str(i)+': client '+str(client_id) + ' sent scaled local model to server')

            # Receive the global model from the server
            global_model = receive_model(client_socket)
            global_weights = copy.deepcopy(global_model.coef_)
            global_b = copy.deepcopy(global_model.intercept_)


        client_socket.close()
        print(f'Client {client_id} disconnected from server...')


def trainLocal(clients_data,comms_round):
    # commence global training loop
    local_models = []
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
        clf = centerSVM(comms_round, X_train, y_train)
        local_models.append(clf)
    return local_models


def trainFederated(X_test,y_test,comms_round,global_model_file,clients_data):
    client_num = len(clients_data)
    client_names = list(clients_data.keys())
    HOST = '127.0.0.1'
    PORT = 12345
    count = 0
    shared_dict = multiprocessing.Manager().dict()

    send_sem = multiprocessing.Semaphore(1)
    recv_sem = multiprocessing.Semaphore(0)

    server = multiprocessing.Process(target=run_server, args=(X_test,y_test,HOST, PORT, comms_round, global_model_file, clients_data,shared_dict))
    server.start()
    time.sleep(5)
    if not server.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []
    for client_id in range(client_num):
        worker = multiprocessing.Process(target=run_client,
            args = (HOST, PORT, comms_round, clients_data, client_names, client_id, count))
        workers.append(worker)
        worker.start()
        print(str(client_id) +" client started")

    for worker in workers:
        worker.join()

    server.terminate()
    for worker in workers:
        worker.terminate()

    return shared_dict['result'],shared_dict['time']