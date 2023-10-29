#!/usr/bin/python
import multiprocessing
import sys
import time

import xgboost as xgb
import xgboost.federated

#federated server
#46

SERVER_KEY = 'server-key.pem'
SERVER_CERT = 'server-cert.pem'
CLIENT_KEY = 'client-key.pem'
CLIENT_CERT = 'client-cert.pem'


def run_server(port: int, world_size: int, with_ssl: bool) -> None:
    if with_ssl:
        xgboost.federated.run_federated_server(port, world_size, SERVER_KEY, SERVER_CERT,
                                               CLIENT_CERT)
    else:
        xgboost.federated.run_federated_server(port, world_size)


if __name__ == '__main__':
    world_size = int(sys.argv[1])
    port = int(sys.argv[2])
    run_server(port, world_size, with_ssl=False)
