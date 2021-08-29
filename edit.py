import numpy as np
from scipy import interpolate

x = np.array([i for i in range(26)])
y = 2 ** (x/5)
find_new_nodes = interpolate.interp1d(x, y, kind = 'cubic', bounds_error=False, fill_value=(0,max(y)))

z = 5 * ( 1 / (2*np.pi)**.5 / 2 ) * np.e ** (-1 * (y-max(y)/2)**2 )
find_new_layers = interpolate.interp1d(y, z, kind = 'cubic', bounds_error=False, fill_value= (0, 0))

w = (-y**1.75) / 2 + 215
find_new_epochs = interpolate.interp1d(y, w, kind = 'cubic', bounds_error=False, fill_value= (215, 0))

x = np.array([i for i in range(29)])
s = 2 * ( 2 ** x )
find_new_l2 = interpolate.interp1d(x, s, kind = 'cubic', bounds_error=False, fill_value= (.02, .5))

def analyze(
    epochs,
    evaluation,
    base_line: object,
    num_of_layers: object,
    nodes_per_layer: object,
    activation_function: object,
    l2_lambda: object,
    dropout_rate: object,
    exceed: object = 1,
) -> object:
    """

    :param epochs:
    :param evaluation:
    :param base_line:
    :param num_of_layers:
    :param nodes_per_layer:
    :param activation_function:
    :param l2_lambda:
    :param dropout_rate:
    :param exceed:
    :return:
    """
    good_enough = False

    test_accuracy = evaluation["Testing accuracy"]
    train_accuracy = evaluation["Training accuracy"]
    goal = exceed * base_line

    if test_accuracy > goal:
        print("NICE NETWORK")
        good_enough = True
    elif train_accuracy < goal:
        print(f"NOT TRAINING HIGH ENOUGH ACCURACY (ONLY {train_accuracy} NOT {goal})")
        percent_error = 100 * (goal - test_accuracy) / test_accuracy
        v = int(find_new_nodes(percent_error))
        nodes_per_layer += v
        print(f"Added {v} nodes per layer (total: {nodes_per_layer})")

        u = int(find_new_layers(v))
        num_of_layers += u
        print(f"Added {u} layers (total: {num_of_layers})")

        t = int(find_new_epochs(v))
        epochs += t
        print(f"Added {t} epochs (total: {epochs})")
    else:
        print("MOST LIKELY OVER-FIT (add dropout and regularization)")
        percent_difference = 100 * (train_accuracy - test_accuracy) / test_accuracy
        print(f"Percent Difference: {percent_difference}")

        v = int( find_new_l2(percent_difference) ) /100
        #l2 ranges from 0 to 1 on a logarithmic scale

        dropout_rate += v
        print(f"Added {v} to Dropout (total: {dropout_rate})")
        #dropout_rate += int(v / 5)

    return (
        epochs,
        num_of_layers,
        nodes_per_layer,
        activation_function,
        l2_lambda,
        dropout_rate,
        good_enough,
    )
