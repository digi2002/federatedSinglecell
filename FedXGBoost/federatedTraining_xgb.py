#federated xgb
#46
from ..utils.federated_client import run_worker
from ..utils.federated_server import run_server
import multiprocessing
import time



with_ssl = False
with_gpu = False
#num_Tree = 0


def trainFederated(X_test,y_test,clients_data,class_num,globalmodelname,port = 8080):
    client_num = len(clients_data)

    server = multiprocessing.Process(target=run_server, args=(port, client_num, with_ssl))
    server.start()
    time.sleep(1)
    if not server.is_alive():
        raise Exception("Error starting Federated Learning server")

    workers = []
    for rank in range(client_num):
        worker = multiprocessing.Process(target=run_worker,
                                         args=(port, client_num, rank, with_ssl, with_gpu, clients_data,class_num,[X_test,y_test],globalmodelname))
        workers.append(worker)
        worker.start()
    for worker in workers:
        worker.join()
    server.terminate()


    #run_server(port, client_num, with_ssl=False)
    #for rank in range(client_num):
    #    run_worker(port = port, world_size = client_num, rank = rank, with_ssl=False, with_gpu=False,clients_data = clients_data)
