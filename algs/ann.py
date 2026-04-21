import numpy as np
import matplotlib.pyplot as plt
import random
from typing import Dict, List, Set, Tuple, Any
from math import exp

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
    if terrain == "hill":
        return [1, 0, 0, 0]
    elif terrain == "swap":
        return [0, 1, 0, 0]
    elif terrain == "forest":
        return [0, 0, 1, 0]
    elif terrain == "plains":
        return [0, 0, 0, 1]

    return None


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
        for proto in data.values():
            blurred = blur(proto)
            blurred[-1] = encode(proto[-1])
            result.append(blurred)
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


def print_model(model, hidden_zs, hidden_activations, output_zs, output_activations):
    n = 1
    for hidden_layer, z, activation in zip(model[0], hidden_zs, hidden_activations):
        print(f"{n},{','.join(str(i) for i in hidden_layer)},NA,{z},{activation}")
        n += 1
    for output_layer, z, activation in zip(model[1], output_zs, output_activations):
        print(f"{n},{','.join(str(i) for i in output_layer)},{z},{activation}")
        n += 1
    return


def feed_forward(model, example):
    example = [1.0] + example  # add bias
    hidden_zs = []
    hidden_activations = []
    for hidden_layer in model[0]:
        z = sum(theta * x for theta, x in zip(hidden_layer, example))
        activation = 1.0 / (1.0 + exp(-1.0 * z))  # 1/ (1 + e^-z)
        hidden_zs.append(z)
        hidden_activations.append(activation)

    output_zs = []
    output_activations = []
    for output_layer in model[1]:
        biased_acts = [1.0] + hidden_activations
        z = sum(theta * x for theta, x in zip(output_layer, biased_acts))
        activation = 1.0 / (1.0 + exp(-1.0 * z))  # 1/ (1 + e^-z)
        output_zs.append(z)
        output_activations.append(activation)
    return hidden_zs, hidden_activations, output_zs, output_activations


def calculate_output_error(output_activations, label):
    deltas = []
    for y_hat, y in zip(output_activations, label):
        delta = y_hat * (1 - y_hat) * (y - y_hat)
        deltas.append(delta)
    return deltas


def calculate_hidden_error(model , hidden_activations, delta_os):
    deltas = []
    for h, y_hat in enumerate(hidden_activations, 1):
        s = 0 
        for output_layer, do in zip(model[1], delta_os):
            e = output_layer[h] * do
            # print(h, y_hat, output_layer[h], do)
            s += e
        delta = y_hat * (1 - y_hat) * s
        deltas.append(delta)

    return deltas


def update_weights(model, example, delta_hs, alpha):
    example = [1.0] + example  # add bias

    for hidden_layer in model[0]:
        for theta, dh, x in zip(hidden_layer, delta_hs, example):
            theta += alpha * dh * x  # update in place?
    
    for output_layer in model[0]:
        for theta, , do in zip(output_layer, delta_os, delta_os):
            theta += alpha * do * y  # update in place?
    

    return


def backprop(model, actual, hidden_activations, output_activations):
    delta_os = calculate_output_error(output_activations, actual)
    print(delta_os)
    delta_hs = calculate_hidden_error(model , hidden_activations, delta_os)
    print(delta_hs)
    # model = update_weights(model, example)
    return model


# def learn_model(data: List[List[Any]], hidden_nodes: int, epsilon: float = 10**-5, alpha: float = 0.1, verbose=False, max_iters: int = 1):
#     hidden_layer = []
#     output_layer  = []

#     model = [
#         hidden_layer,
#         output_layer
#     ]
#     iterations = 0
#     prev_error = 0
#     while iterations < max_iters:

#         for example in data:
#             feed_forward(model, example)
#             delta_o = calculate_output_error(model, example)
#             delta_h = calculate_hidden_error(model, example)
#             model = update_weights(model, example)

#         if abs(delta_o - prev_error) <= epsilon:
#           return model

#     return None


def test():
    input_layer = [0.52, -0.97]
    actual = [1, 0]
    hidden_layer = [[0.01, 0.26, -0.42], [-0.05, 0.78, 0.19], [0.42, -0.23, 0.37]]
    output_layer = [[0.2, 0.61, 0.12, -0.9], [0.3, 0.28, -0.34, 0.10]]
    model = [hidden_layer, output_layer]
    hidden_zs, hidden_activations, output_zs, output_activations = feed_forward(model, input_layer)
    print_model(model, hidden_zs, hidden_activations, output_zs, output_activations)
    print()
    backprop(model, actual, hidden_activations, output_activations)


test()
