from sklearn.linear_model import SGDClassifier
import xgboost as xgb


def centerSVM(comms_round,X_train, y_train):
    center_clf = SGDClassifier(random_state=0, loss='hinge', max_iter=comms_round)
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