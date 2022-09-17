import random


def create_clients(feature_list, label_list, num_clients=10, initial='clients'):
    ''' return: a dictionary with keys clients' names and value as
                data shards - tuple of images and label lists.
        args:
            image_list: a list of numpy arrays of training images
            label_list:a list of binarized labels for each image
            num_client: number of fedrated members (clients)
            initials: the clients'name prefix, e.g, clients_1

    '''
    # create a list of client names
    client_names = ['{}_{}'.format(initial, i+ 1) for i in range(num_clients)]

    # randomize the data
    data = list(zip(feature_list, label_list))
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))}


if __name__=="__main__":
    feature_list = [1,2,3,4,5,2]
    label_list = [0,0,0,0,1,0]
    m = create_clients(feature_list, label_list, num_clients=3, initial='clients')
    print('d')
