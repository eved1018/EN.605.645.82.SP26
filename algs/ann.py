import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Set, Tuple, Any, NamedTuple
from math import exp, inf

clean_data = {
    "plains": [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, "plains"]],
    "forest": [
        [0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, "forest"],
        [1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, "forest"],
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, "forest"],
    ],
    "hills": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, "hills"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, "hills"],
    ],
    "swamp": [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, "swamp"],
    ],
}


def blur(data) -> List[Any]:
    def apply_noise(value):
        if value < 0.5:
            v = random.gauss(0.10, 0.05)
            if v < 0.0:
                return 0.0
            if v > 0.75:
                return 0.75
            return v
        else:
            v = random.gauss(0.90, 0.10)
            if v < 0.25:
                return 0.25
            if v > 1.00:
                return 1.00
            return v

    noisy_readings = [apply_noise(v) for v in data[0:-1]]
    return noisy_readings + [data[-1]]


# view_sensor_image( blur( clean_data["swamp"][0]))


def encode(terrain: str) -> List[int] | None:
    terrain2bin = {
        "hills": [1, 0, 0, 0],
        "swamp": [0, 1, 0, 0],
        "forest": [0, 0, 1, 0],
        "plains": [0, 0, 0, 1],
    }

    return terrain2bin.get(terrain, None)


def generate_data(data: Dict[str, List[Any]], n: int) -> List[List[Any]]:
    """
    Generates an endless supply of blurred data from a collection of terrain prototypes.

    * `data`: Dict[Str, List[Any]] - a Dictionary of "clean" prototypes for each landscape type.
    * `n`: Int - the number of blurred examples of each terrain type to return.

    returns

    * List[[List[Any]] - a List of Lists. Each individual List is a blurred example of a terrain type, generated from the prototype.
    """
    result = []
    for _ in range(n):
        for terrain, prototypes in data.items():
            proto = random.choice(prototypes)
            row = blur(proto)
            row[-1] = encode(terrain)
            result.append(row)
    return result


"""
create a network with the required number of input, hidden and  
 output nodes. This mostly amounts to creating a List of Lists of 
 thetas for the hidden and output layers. Don't forget biases for  
 every node.  
2 initialize all thetas to small random values (0..1) or (-1, 1). 
3 until termination  
4     for each point in the training set  
          # feed forward step  
5         calculate output of every node in the network.  
          # back prop step  
6         calculate delta_o for every output node  
7         calculate delta_h for every hidden node  
8         update all of the thetas     


"""

Node = List[float]
Layer = List[Node]
Model = NamedTuple("Model", [("hidden", List[Layer]), ("output", Layer)])


def make_network(n_inputs: int, n_hidden_layers: int, n_outputs: int, weight_range: float = 0.5):
    def rand_weights(n: int) -> List[float]:
        return [random.uniform(-weight_range, weight_range) for _ in range(n)]

    hidden_layer = [rand_weights(n_inputs + 1) for _ in range(n_hidden_layers)]
    output_layer = [rand_weights(n_hidden_layers + 1) for _ in range(n_outputs)]
    return Model([hidden_layer], output_layer)


def print_model(model: Model, hidden_zs: List[List[float]], hidden_activations: List[List[float]], output_zs: List[float], output_activations: List[float]):
    n = 1
    for hidden_layer, zs, acts in zip(model[0], hidden_zs, hidden_activations):
        print(f"{n},weights={hidden_layer},NA,zs={zs},acts={acts}")
        n += 1
    for node, z, act in zip(model[1], output_zs, output_activations):
        print(f"{n},weights={node},z={z:.6f},act={act:.6f}")
        n += 1
    return


def feed_forward(model: Model, example: List[float]) -> Tuple[List[List[float]], List[List[float]], List[float], List[float]]:
    example = [1.0] + example  # add bias
    hidden_zs = []
    hidden_activations = []
    for hidden_layer in model.hidden:
        layer_activation = []
        layer_zs = []
        for node in hidden_layer:
            z = sum(theta * x for theta, x in zip(node, example))
            activation = 1.0 / (1.0 + exp(-1.0 * z))  # 1/ (1 + e^-z)
            layer_zs.append(z)
            layer_activation.append(activation)

        hidden_zs.append(layer_zs)
        hidden_activations.append(layer_activation)

    output_zs = []
    output_activations = []
    for output_layer in model.output:
        biased_acts = [1.0] + hidden_activations[-1]
        z = sum(theta * x for theta, x in zip(output_layer, biased_acts))
        activation = 1.0 / (1.0 + exp(-1.0 * z))  # 1/ (1 + e^-z)
        output_zs.append(z)
        output_activations.append(activation)
    return hidden_zs, hidden_activations, output_zs, output_activations


def calculate_output_error(output_activations: List[float], label: List[float]) -> List[float]:
    deltas = []
    for y_hat, y in zip(output_activations, label):
        delta = y_hat * (1 - y_hat) * (y - y_hat)
        deltas.append(delta)
    return deltas


def calculate_hidden_error(model: Model, hidden_activations: List[List[float]], delta_os: List[float]) -> List[List[float]]:
    deltas = []
    for layer_activations in hidden_activations:
        layer_deltas = []
        for h, y_hat in enumerate(layer_activations):
            s = 0
            for output_layer, do in zip(model.output, delta_os):
                e = output_layer[h + 1] * do  # +1 to skip bias
                s += e
            delta = y_hat * (1 - y_hat) * s
            layer_deltas.append(delta)
        deltas.append(layer_deltas)
    return deltas


def update_weights(model: Model, example: List[float], delta_hs: List[List[float]], hidden_activations: List[List[float]], delta_os: List[float], alpha: float) -> Model:
    biased_input = [1.0] + example

    # Hidden layers
    for layer_idx in range(len(model.hidden)):
        layer_deltas = delta_hs[layer_idx]  # one scalar per node
        for n, dh in enumerate(layer_deltas):
            for w, x in enumerate(biased_input):
                model[0][layer_idx][n][w] += alpha * dh * x

    # Output layer
    biased_hidden = [1.0] + hidden_activations[-1]  # bias + hidden activations
    for n, do in enumerate(delta_os):
        for w, x in enumerate(biased_hidden):
            model[1][n][w] += alpha * do * x

    return model


def backprop(
    model: Model, example: List[Any], actual: List[Any], hidden_activations: List[List[float]], output_activations: List[float], alpha: float
) -> Tuple[List[float], List[List[float]]]:
    delta_os = calculate_output_error(output_activations, actual)
    delta_hs = calculate_hidden_error(model, hidden_activations, delta_os)
    model = update_weights(model, example, delta_hs, hidden_activations, delta_os, alpha)
    return delta_os, delta_hs


def learn_model(data: List[List[Any]], n_hidden_layers: int, epsilon: float = 10**-5, alpha: float = 0.01, verbose: bool = True, max_iters: int = 10000, print_freq: int = 1000):
    n_inputs = len(data[0]) - 1
    n_outputs = len(data[0][-1])
    model = make_network(n_inputs, n_hidden_layers, n_outputs)
    iterations = 0
    prev_error = 0
    while iterations < max_iters:
        error = 0.0
        for example in data:
            features, label = example[:-1], example[-1]
            hidden_zs, hidden_activations, output_zs, output_activations = feed_forward(model, example)
            backprop(model, features, label, hidden_activations, output_activations, alpha)
            error += sum((y - o) ** 2 for y, o in zip(label, output_activations))

        error = error / len(data)
        if error > prev_error:
            alpha /= 10

        if verbose and (iterations % print_freq == 0):
            print(f"{iterations}\t{error:.6f}")

        if abs(error - prev_error) < epsilon:
            return model
        prev_error = error
        iterations += 1

    return None


def apply_model(model: Model, test_data: List[List[Any]], labeled=False):
    """

    `apply_model` takes the ANN (the model) and either labeled or unlabeled data.
    If the data is unlabeled, it will return predictions for each observation as a List of Tuples
    of the inferred value (0 or 1) and the actual probability (so something like (1, 0.73) or (0, 0.19)
    so you have [(0, 0.30), (1, 0.98), (0, 0.87), (0, 0.12)]. Note that unlike the logistic regression,
    the threshold for 1 is not 0.5 but which value is largest (0.98 in this case).

    If the data is labeled, you will return a List of List of Tuples of the actual value (0 or 1)
    and the predicted value (0 or 1). For a single data point, you'll have the pairs of actual values
    [(0, 1), (0, 0), (0, 0), (1, 0)] is a misclassification and [(0, 0), (0, 0), (1, 1), (0, 0)] will
    be a correct classification. Then you have a List of *those*, one for each observation.
    """
    result = []
    for example in test_data:
        features = example[:-1] if labeled else example
        _, _, _, output_activations = feed_forward(model, features)
        max_node, max_val = None, -inf
        for node, y_hat in enumerate(output_activations):
            if y_hat > max_val:
                max_val = y_hat
                max_node = node

        row = []
        for node, (y, y_hat) in enumerate(zip(example[-1], output_activations)):
            prediction = 1 if node == max_node else 0
            if labeled:
                row.append((y, prediction))
            else:
                row.append((prediction, y_hat))
        result.append(row)
    return result


def evaluate(result):
    tp = 0
    for row in result:
        if (1, 1) in row:
            tp += 1
    print(tp / len(result))
    return


def test1():
    input_layer = [0.52, -0.97]
    actual = [1, 0]
    hidden_layer = [[0.01, 0.26, -0.42], [-0.05, 0.78, 0.19], [0.42, -0.23, 0.37]]
    output_layer = [[0.2, 0.61, 0.12, -0.9], [0.3, 0.28, -0.34, 0.10]]
    model = Model([hidden_layer], output_layer)
    alpha = 0.01
    hidden_zs, hidden_activations, output_zs, output_activations = feed_forward(model, input_layer)
    print_model(model, hidden_zs, hidden_activations, output_zs, output_activations)
    print()
    backprop(model, input_layer, actual, hidden_activations, output_activations, alpha)
    print_model(model, hidden_zs, hidden_activations, output_zs, output_activations)


def test2(clean_data):
    dataset = generate_data(clean_data, 100)
    model = learn_model(dataset, 1)
    test_set = generate_data(clean_data, 20)
    result = apply_model(model, test_set, True)
    for r in result:
        print(r)
    evaluate(result)


# test1()
test2(clean_data)
