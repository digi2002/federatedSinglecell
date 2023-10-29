#save
from sklearn.linear_model import SGDClassifier
#import xgboost as xgb


def centerSVM(comms_round,X_train, y_train):
    #0.0001
    center_clf = SGDClassifier(random_state=0, loss='hinge', learning_rate = 'constant', eta0 = 0.000001,alpha = 0.5,
                               class_weight = 'balanced',max_iter=comms_round)
    center_clf.fit(X_train, y_train)
    return center_clf


def centerXGB(treenum,X_train, y_train):
    print('start training')
    D_train = xgb.DMatrix(X_train, label=y_train)
    param = {
        'eta': 0.3,
        'max_depth': 3,
        'objective': 'multi:softprob',
        'num_class': len(set(y_train))}
    steps = treenum  # The number of training iterations
    model = xgb.train(param, D_train, steps)
    return model