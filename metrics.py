from sklearn.metrics import classification_report
def doIt(X_train, Y_train, X_valid, Y_valid, transformation, clf):
    clf.fit(transformation(X_train), Y_train)
    pred1 = clf.predict(transformation(X_valid))
    print str(f1_score(pred1, Y_valid))
    print str(classification_report(Y_valid, pred1))



