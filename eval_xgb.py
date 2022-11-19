from sklearn.metrics import accuracy_score,f1_score
import numpy as np
from sklearn.metrics import f1_score,precision_score,recall_score
import xgboost as xgb

def xgbinfer(modelname, X_test,y_test):
    D_test = xgb.DMatrix(X_test, label=y_test)
    model = xgb.Booster(model_file=modelname)
    preds = model.predict(D_test)
    best_preds = np.asarray([np.argmax(line) for line in preds])

    acc=accuracy_score(y_test, best_preds)
    f1 = f1_score(y_test, best_preds, average='weighted')
    recall = recall_score(y_test, best_preds, average='weighted')
    precision = precision_score(y_test, best_preds, average='weighted')
    return acc,f1

def testSet(clients,global_modelname, local_modelname, center_modelname,X_test,y_test):
    global_acc, global_f1 = xgbinfer(global_modelname, X_test,y_test)
    local_acc,local_f1 = evalLocals(clients,local_modelname, X_test,y_test)
    center_acc,center_f1 = xgbinfer(center_modelname, X_test,y_test)

    dict = {'global_acc':global_acc,
            'global_f1':global_f1,
            'local_acc':local_acc,
            'local_f1':local_f1,
            'center_acc':center_acc,
            'center_f1':center_f1}
    return dict

def evalLocals(clients,modelname, X_test,y_test):
    local_accs = []
    local_f1s = []
    for rank in range(len(clients)):
        localmodelname = modelname + '_' + str(rank)
        acc, f1 = xgbinfer(localmodelname, X_test,y_test)
        local_accs.append(acc)
        local_f1s.append(f1)
    return np.average(np.array(local_accs)),np.average(np.array(local_f1s))
