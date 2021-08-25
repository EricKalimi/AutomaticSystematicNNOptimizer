import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical


def setup_data(file_name, target, test_s=0.2):
    """

    :param file_name:
    :param target:
    :param test_s:
    :return:
    """
    df = pd.read_csv(file_name).dropna()

    x = df.drop(columns=[target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_s)

    x_train = normalize(x_train)
    x_test = normalize(x_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return x, y, x_train, y_train, x_test, y_test
