import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import regularizers
from sklearn.metrics import accuracy_score


def build(num_of_layers, nodes_per_layer, activation_function, L2_lambda, Dropout_rate, mean=0, standard_deviation=1):
    tf.keras.backend.clear_session()

    model = Sequential()

    random = tf.keras.initializers.RandomNormal(mean=mean, stddev=standard_deviation)

    from main import num_input_neurons, num_output_neurons

    model.add(Dense(num_input_neurons))
    for i in range(num_of_layers):
        model.add(Dense(units=nodes_per_layer, activation=activation_function, kernel_initializer=random,kernel_regularizer=regularizers.l2(L2_lambda)))
        model.add(Dropout(Dropout_rate))

    model.add(Dense(units=num_output_neurons, activation='softmax', kernel_initializer=random))

    return model


class CustomCallback(tf.keras.callbacks.Callback):
    min_accuracy=.95
    counter = 0
    previous_accuracy = 0
    good_counter = 0

    def on_epoch_end(self, epoch, logs=None):
        if logs.get("accuracy") > self.min_accuracy:
            self.good_counter += 1
            if self.good_counter > 10:
                print(f"Accuracy consistently over {100 * self.min_accuracy}%, training terminated.")
                self.model.stop_training = True
        else:
            self.good_counter = 0

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


def train(model, Index_of_Network, X_train, y_train, epochs=1000, loss_function='categorical_crossentropy',
          optimzr="rmsprop", metrics=['accuracy']):
    model.compile(loss=loss_function, optimizer=optimzr, metrics=metrics)
    model.fit(X_train, y_train, epochs=epochs, batch_size=len(X_train), callbacks=[CustomCallback()])

    model.save(f'/networks/saved_model{Index_of_Network}/')
    return model


def evaluateNN(model, Index_of_Network, X_train, y_train, X_test, y_test):
    y_train1d = [list(x).index(max(x)) for x in y_train]
    y_test1d = [list(x).index(max(x)) for x in y_test]

    y_hat1 = [list(x).index(max(x)) for x in model.predict(X_train)]
    training_accuracy = 100 * accuracy_score(y_train1d, y_hat1)
    print("NN Training Accuracy:", training_accuracy)
    cm = tf.math.confusion_matrix(y_train1d, y_hat1)
    print(cm)

    y_hat = [list(x).index(max(x)) for x in model.predict(X_test)]
    testing_accuracy = 100 * accuracy_score(y_test1d, y_hat)
    print("NN Testing Accuracy:", testing_accuracy)
    cm1 = tf.math.confusion_matrix(y_test1d, y_hat)
    print(cm1)

    from main import ALL_NETWORKS

    ALL_NETWORKS.append({'Testing accuracy': testing_accuracy, 'Training accuracy': training_accuracy})

    return ALL_NETWORKS[-1]