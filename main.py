from Data import setup_data
from Network import build, CustomCallback, train, evaluateNN
from Edit import analyze

import pandas as pd

file_name = 'train.csv'
target = 'label'

X, y, X_train, y_train, X_test, y_test = setup_data(file_name,target)

Index_of_Network = 0
ALL_NETWORKS = []
List_of_models = []
num_output_neurons = len(pd.unique(y))
num_input_neurons = X.shape[1]
num_of_layers = 10
nodes_per_layer = 100
activation_function = "sigmoid"
L2_lambda = 0
Dropout_rate = 0



def create_network(num_of_layers, nodes_per_layer, activation_function, L2_lambda, Dropout_rate):
  model = build(num_of_layers, nodes_per_layer, activation_function, L2_lambda, Dropout_rate)
  model = train(model, Index_of_Network, X_train, y_train)
  return evaluateNN(model, Index_of_Network, X_train, y_train, X_test, y_test)

for i in range(3):
  evaluation = create_network(num_of_layers,nodes_per_layer,activation_function,L2_lambda,Dropout_rate)
  num_of_layers,nodes_per_layer,activation_function,L2_lambda,Dropout_rate = analyze(evaluation, num_of_layers,nodes_per_layer,activation_function,L2_lambda,Dropout_rate)
