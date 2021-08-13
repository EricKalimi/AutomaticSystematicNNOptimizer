import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import normalize, to_categorical


def setup_data(file_name, target, test_s=.2):
    df = pd.read_csv(file_name).dropna()

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_s)

    X_train = normalize(X_train)
    X_test = normalize(X_test)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X, y, X_train, y_train, X_test, y_test