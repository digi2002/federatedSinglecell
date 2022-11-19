from sklearn.metrics import accuracy_score,f1_score
import numpy as np

def testSet(global_model,local_models,center_model,X_test,y_test):
    global_acc, global_f1 = testModel(X_test,y_test,global_model)
    local_acc,local_f1 = evalLocals(X_test,y_test,local_models)
    center_acc,center_f1 = testModel(X_test,y_test,center_model)
    dict = {'global_acc':global_acc,
            'global_f1':global_f1,
            'local_acc':local_acc,
            'local_f1':local_f1,
            'center_acc':center_acc,
            'center_f1':center_f1}
    return dict



def testModel(X_test,y_test,clf):
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_pred, y_test)
    f1 = f1_score(y_pred, y_test, average='weighted')
    return acc,f1

def evalLocals(X_test,y_test,clfs):
    local_accs = []
    local_f1s = []
    for clf in clfs:
        acc, f1 = testModel(X_test, y_test, clf)
        local_accs.append(acc)
        local_f1s.append(f1)
    return np.average(np.array(local_accs)),np.average(np.array(local_f1s))
