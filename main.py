from data import setup_data
from network import build, CustomCallback, train, evaluateNN
from base import find_base
from edit import analyze

import pandas as pd

file_name = "train.csv"
target = "label"
x, y, x_train, y_train, x_test, y_test = setup_data(file_name, target)


Index_of_Network = 0
ALL_NETWORKS = []
List_of_models = []
num_output_neurons = len(pd.unique(y))
num_input_neurons = x.shape[1]
num_of_layers = 10
nodes_per_layer = 50
epochs = 100
activation_function = "sigmoid"
l2_lambda = 0
dropout_rate = 0

base_line = find_base(x_train, y_train, x_test, y_test)
exceed_base_model = 1.08
if base_line * exceed_base_model >= 99.9:
    exceed_base_model = 99.9


def create_network(
    epochs, num_of_layers, nodes_per_layer, activation_function, l2_lambda, dropout_rate
):
    """

    :param epochs:
    :param num_of_layers:
    :param nodes_per_layer:
    :param activation_function:
    :param l2_lambda:
    :param dropout_rate:
    :return:
    """
    model = build(
        num_of_layers, nodes_per_layer, activation_function, l2_lambda, dropout_rate
    )
    model = train(model, Index_of_Network, x_train, y_train, epochs)
    model.summary()
    return evaluateNN(model, Index_of_Network, x_train, y_train, x_test, y_test)


for i in range(5):
    evaluation = create_network(
        epochs,
        num_of_layers,
        nodes_per_layer,
        activation_function,
        l2_lambda,
        dropout_rate,
    )
    (
        epochs,
        num_of_layers,
        nodes_per_layer,
        activation_function,
        l2_lambda,
        dropout_rate,
        good_enough,
    ) = analyze(
        epochs,
        evaluation,
        base_line,
        num_of_layers,
        nodes_per_layer,
        activation_function,
        l2_lambda,
        dropout_rate,
        exceed_base_model,
    )
    print(f'This is the {i}th network')
    if good_enough:
        print("This network is complete")
        break
