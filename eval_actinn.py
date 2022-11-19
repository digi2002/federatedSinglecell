from sklearn.metrics import accuracy_score,f1_score
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
from actinn import predict

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
    return acc,f1

def testSet(global_parameters, local_parameters, center_parameters,label_to_type_dict,test_set,y_test):
    global_acc, global_f1 = actininfer(test_set, y_test, global_parameters, label_to_type_dict)
    local_acc,local_f1 = evalLocals(test_set,y_test,local_parameters,label_to_type_dict)
    center_acc,center_f1 = actininfer(test_set, y_test, center_parameters, label_to_type_dict)

    dict = {'global_acc':global_acc,
            'global_f1':global_f1,
            'local_acc':local_acc,
            'local_f1':local_f1,
            'center_acc':center_acc,
            'center_f1':center_f1}
    return dict



def evalLocals(test_set,y_test,local_parameters,label_to_type_dict):
    local_accs = []
    local_f1s = []
    for parameters in local_parameters:
        acc, f1 = actininfer(test_set,y_test,parameters,label_to_type_dict)
        local_accs.append(acc)
        local_f1s.append(f1)
    return np.average(np.array(local_accs)),np.average(np.array(local_f1s))
