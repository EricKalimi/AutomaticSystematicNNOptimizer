def analyze(evaluation, num_of_layers, nodes_per_layer, activation_function, L2_lambda, Dropout_rate):
    test_accuracy = evaluation['Testing accuracy']
    train_accuracy = evaluation['Training accuracy']

    if train_accuracy * .95 > test_accuracy:
        print("MOST LIKELY OVERFIT")
        num_of_layers += 1
        nodes_per_layer += 5
    elif test_accuracy < 99:
        print("NOT TESTING HIGH ENOUGH ACCURACY")
        num_of_layers += 1
        nodes_per_layer += 5
    else: print("This is good")

    return num_of_layers, nodes_per_layer, activation_function, L2_lambda, Dropout_rate