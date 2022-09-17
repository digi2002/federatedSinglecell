import pandas as pd
import numpy as np


def read_counts2tpm(df, sample_name):
    """
    convert read counts to TPM (transcripts per million)
    :param df: a dataFrame contains the result coming from featureCounts
    :param sample_name: a list, all sample names, same as the result of featureCounts
    :return: TPM
    """
    result = df
    sample_reads = result.loc[:, sample_name].copy()
    gene_len = result.loc[:, ['Length']]
    rate = sample_reads.values / gene_len.values
    tpm = rate / np.sum(rate, axis=0).reshape(1, -1) * 1e6
    return pd.DataFrame(data=tpm, columns=sample_name)

def readTrain(filename):
    df = pd.read_csv(filename)
    data = df.to_numpy()
    X = data[:, 1:].astype(float)
    y = data[:, 0]
    return X,y


def readCounts2CPM(X):
    new_X = []
    for sample in X:
        tpm = sample / np.sum(sample)*1e6
        tpm_log = np.log2(tpm + 1)
        new_X.append(tpm_log)
    return np.array(new_X)

def removeZerogene(X):
    new_X_t = []
    X_t = np.transpose(X)
    for gene in X_t:
        if np.sum(gene)!=0:
            new_X_t.append(gene)
    new_X_t = np.array(new_X_t)
    return np.transpose(new_X_t)




if __name__=="__main__":

    X=[[1,2,0,1],[2,3,0,0]]
    X = removeZerogene(X)

    trainfile = './singlecell/train.csv'

    comms_round = 10

    X_train, y_train = readTrain(trainfile)
    readCounts2CPM(X_train)

