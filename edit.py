scale = 0.2


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
    made_bigger = False

    test_accuracy = evaluation["Testing accuracy"]
    train_accuracy = evaluation["Training accuracy"]
    goal = exceed * base_line

    global scale

    if test_accuracy > goal:
        print("NICE NETWORK")
        good_enough = True
    elif not made_bigger and train_accuracy > goal:
        print("MOST LIKELY OVER-FIT (make network smaller)")
        v = round((train_accuracy - test_acuracy) * scale) + 1

        if v > 10:
            v = 10
        num_of_layers -= v
        nodes_per_layer -= 3 * v
        epochs -= 25 * v
    elif train_accuracy > goal:
        print("MOST LIKELY OVER-FIT (add dropout and regularization)")
        v = round((train_accuracy - test_accuracy) * scale)

        if v > 1:
            v = 1
        l2_lambda += v / 5
        dropout_rate += v / 5

    else:
        print(f"NOT TESTING HIGH ENOUGH ACCURACY (ONLY {test_accuracy} NOT {goal})")
        v = round((goal - test_accuracy) * scale) + 1
        if v > 10:
            v = 10
        num_of_layers += v
        nodes_per_layer += 3 * v
        epochs += 50 * v

    return (
        epochs,
        num_of_layers,
        nodes_per_layer,
        activation_function,
        l2_lambda,
        dropout_rate,
        good_enough,
    )
