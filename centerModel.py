from sklearn.linear_model import SGDClassifier


def centerSVM(comms_round,X_train, y_train):
    center_clf = SGDClassifier(random_state=0, loss='hinge', max_iter=comms_round)
    center_clf.fit(X_train, y_train)
    return center_clf




