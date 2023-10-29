#save
import random
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from actinn import scale_sets,type_to_label_dict1,convert_type_to_label,one_hot_matrix,model,predict
import pandas as pd
from sklearn.metrics import f1_score,precision_score,recall_score
import time
import socket
import json
import multiprocessing
import pickle

BUFFER_SIZE = 4096
def receive_model(sock):
    data = sock.recv(4)
    stream_len = int.from_bytes(data, byteorder='big')
    print(f"Stream length: {stream_len}")
    # Receive the stream itself
    chunks = []
    bytes_recd = 0
    while bytes_recd < stream_len:
        chunk = sock.recv(min(stream_len - bytes_recd, BUFFER_SIZE))
        if not chunk:
            raise RuntimeError("Socket connection broken")
        chunks.append(chunk)
        bytes_recd += len(chunk)
        #print(str(bytes_recd) + ' _ ' + str(stream_len) + ' received')
    stream = b''.join(chunks)
    # Deserialize the stream
    model_params = pickle.loads(stream)
    return model_params



def send_model(sock,data):
    stream = pickle.dumps(data)
    stream_len = len(stream)
    sock.sendall(stream_len.to_bytes(4, byteorder='big'))
    # Send the stream itself in chunks
    bytes_sent = 0
    while bytes_sent < stream_len:
        chunk = stream[bytes_sent: bytes_sent + BUFFER_SIZE]
        sock.sendall(chunk)
        bytes_sent += len(chunk)
        #print(str(bytes_sent) + " _ " +str(stream_len) +' sending')



def send_model1(sock,model):
    json_data = json.dumps(model)
    # Send the JSON string to the server
    sock.sendall(json_data.encode('utf-8'))
    print('send successfully')

def receive_model1(sock):
    # Receive the JSON string from the client
    json_data = sock.recv(1024).decode('utf-8')
    # Decode the JSON string into a dictionary
    data = json.loads(json_data)
    return data


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
    #print("Cell Types in training set:", type_to_label_dict)
    #print("# Cells in training set:", len(train_label))
    #print("# Genes in training set:", str(train_set.shape[0]))
    train_label = convert_type_to_label(train_label, type_to_label_dict)
    train_label = one_hot_matrix(train_label, nt)
    parameters = model(train_set, train_label,parameters=parameters,num_epochs = epoch)
    return parameters,label_to_type_dict



def run_server(X_test,y_test,HOST,PORT,comms_round,global_parameters,local_parameters,clients_data,shared_dict,send_semaphores, receiv_semaphores):
    global_start_time = time.time()
    client_num = len(clients_data.keys())
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
            send_model(client_socket,global_parameters)
            print('intial gobal model have been sent to client ')

        # Start the training loop
        for round in range(comms_round):
            parameters_list = {"W1": [],
                               "b1": [],
                               "W2": [],
                               "b2": [],
                               "W3": [],
                               "b3": [],
                               "W4": [],
                               "b4": []}
            for client_id in range(client_num):
                #receiv_semaphores[client_id].acquire()

                client_socket = client_sockets[client_id]
                # Receive the updated model weights from client i
                local_parameters[client_id] = receive_model(client_socket)
                print('round '+str(round)+' server received scaled local model from client '+str(client_id))
                scaling_factor = 1/client_num
                for para_key in local_parameters[client_id].keys():
                    scaled_para = scale_model_weights(local_parameters[client_id][para_key],scaling_factor)
                    parameters_list[para_key].append(scaled_para)

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


            for client_id in range(client_num):
                client_socket = client_sockets[client_id]
                send_model(client_socket,global_parameters)
                print('round '+str(round)+' server send updated global model to client '+str(client_id))

            #test_pred = global_model.predict(X_test)
            #print('round ' + str(round) + ' global performance')
            #print(accuracy_score(y_test, test_pred))

            shared_dict['result'] = global_parameters
            global_end_time = time.time()
            global_time = global_end_time - global_start_time
            shared_dict['time'] = global_time

            #send_semaphores[client_id].release()

        server_socket.close()


def run_client(HOST,PORT,comms_round,clients_data, client_names, client_id,send_semaphore, receiv_semaphore):
    print('in client '+str(client_id))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect((HOST, PORT))
        # Receive the initial global model from the server
        global_parameters = receive_model(client_socket)
        print('global_para received in client '+ str(client_id))
        print(global_parameters)
        for i in range(comms_round):
            # Train the local model
            client = client_names[client_id]
            client_data = clients_data[client]
            X_train, y_train = list(map(list, zip(*client_data)))
            train_set = np.transpose(X_train)

            # set local model weight to the weight of the global model and fit local model with client's data
            local_parameters, type_to_label_dict = actinnmodel(train_set, y_train, global_parameters, epoch=1)

            #send_semaphore.acquire()
            send_model(client_socket, local_parameters)
            print('round ' + str(i) + ': client ' + str(client_id) + ' sent original local model to server')
            # Receive the global model from the server
            global_parameters = receive_model(client_socket)
            print('round ' + str(i) + ': client ' + str(client_id) + ' sent original local model to server')
            #receiv_semaphore.release()

        client_socket.close()
        print(f'Client {client_id} disconnected from server...')

def trainFederated(X_test,y_test,comms_round,global_parameters,local_parameters,clients_data):
    client_num = len(clients_data)
    client_names = list(clients_data.keys())
    HOST = '127.0.0.1'
    PORT = 12345
    shared_dict = multiprocessing.Manager().dict()
    # Create a server semaphore and client semaphores
    send_semaphores = [multiprocessing.Semaphore(1) for i in range(client_num)]
    receive_semaphores = [multiprocessing.Semaphore(0) for i in range(client_num)]


    server = multiprocessing.Process(target=run_server,
                                     args=(X_test,y_test,HOST, PORT, comms_round, global_parameters,local_parameters, clients_data,shared_dict,send_semaphores, receive_semaphores))
    server.start()
    time.sleep(5)
    if not server.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []

    for client_id in range(client_num):
        worker = multiprocessing.Process(target=run_client,
            args = (HOST, PORT, comms_round, clients_data, client_names, client_id,send_semaphores[client_id], receive_semaphores[client_id]))
        workers.append(worker)
        worker.start()
        print(str(client_id) +" client started")

    for worker in workers:
        worker.join()

    server.terminate()
    for worker in workers:
        worker.terminate()

    return shared_dict['result'],shared_dict['time']


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




