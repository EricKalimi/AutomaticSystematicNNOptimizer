import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score


def build(
    num_of_layers,
    nodes_per_layer,
    activation_function,
    l2_lambda,
    dropout_rate,
    mean=0,
    standard_deviation=1,
):
    """

    :param num_of_layers:
    :param nodes_per_layer:
    :param activation_function:
    :param l2_lambda:
    :param dropout_rate:
    :param mean:
    :param standard_deviation:
    :return:
    """
    tf.keras.backend.clear_session()

    model = Sequential()

    random = tf.keras.initializers.RandomNormal(mean=mean, stddev=standard_deviation)

    from main import num_input_neurons, num_output_neurons

    model.add(Dense(num_input_neurons))
    for i in range(num_of_layers):
        model.add(
            Dense(
                units=nodes_per_layer,
                activation=activation_function,
                kernel_initializer=random,
                kernel_regularizer=regularizers.l2(l2_lambda),
            )
        )
        model.add(Dropout(dropout_rate))

    model.add(
        Dense(units=num_output_neurons, activation="softmax", kernel_initializer=random)
    )

    return model


class CustomCallback(tf.keras.callbacks.Callback):

    counter = 0
    previous_accuracy = 0

    def on_epoch_end(self, epoch, logs=None):

        if epoch == 1:
            pass
        elif self.previous_accuracy == logs.get("accuracy"):
            self.counter += 1
        else:
            self.counter = 0

        if self.counter == 10:
            print(f"This network reached a maximum accuracy at {logs.get('accuracy')} ")
            self.model.stop_training = True

        self.previous_accuracy = logs.get("accuracy")


def train(
    model,
    index_of_network,
    x_train,
    y_train,
    epochs,
    loss_function="categorical_crossentropy",
    optimzr="rmsprop",
    metrics=["accuracy"],
):
    """

    :param model:
    :param index_of_network:
    :param x_train:
    :param y_train:
    :param epochs:
    :param loss_function:
    :param optimzr:
    :param metrics:
    :return:
    """
    model.compile(loss=loss_function, optimizer=optimzr, metrics=metrics)
    model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=len(x_train),
        callbacks=[CustomCallback()],
    )

    model.save(f"/networks/saved_model{index_of_network}/")
    return model


def evaluateNN(model, x_train, y_train, x_test, y_test):
    """

    :param model:
    :param x_train:
    :param y_train:
    :param x_test:
    :param y_test:
    :return:
    """
    y_train1d = [list(x).index(max(x)) for x in y_train]
    y_test1d = [list(x).index(max(x)) for x in y_test]

    y_hat1 = [list(x).index(max(x)) for x in model.predict(x_train)]
    training_accuracy = 100 * accuracy_score(y_train1d, y_hat1)
    print("NN Training Accuracy:", training_accuracy)

    y_hat = [list(x).index(max(x)) for x in model.predict(x_test)]
    testing_accuracy = 100 * accuracy_score(y_test1d, y_hat)
    print("NN Testing Accuracy:", testing_accuracy)

    from main import ALL_NETWORKS

    ALL_NETWORKS.append(
        {"Testing accuracy": testing_accuracy, "Training accuracy": training_accuracy}
    )

    return ALL_NETWORKS[-1]
