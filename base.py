from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def find_base(x_train, y_train, x_test, y_test):
    """

    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)

    prediction = rf.predict(x_test)
    return 100 * accuracy_score(y_test, prediction)
