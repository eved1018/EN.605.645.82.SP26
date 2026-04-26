import random
from typing import Dict, List, Set, Tuple, Any, NamedTuple
from math import exp, inf, log10

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
            theta_o = random.gauss(0.10, 0.05)
            if theta_o < 0.0:
                return 0.0
            if theta_o > 0.75:
                return 0.75
            return theta_o
        else:
            theta_o = random.gauss(0.90, 0.10)
            if theta_o < 0.25:
                return 0.25
            if theta_o > 1.00:
                return 1.00
            return theta_o

    noisy_readings = [apply_noise(theta_o) for theta_o in data[0:-1]]
    return noisy_readings + [data[-1]]


def encode(terrain: str) -> List[int] | None:
    terrain2bin = {
        "hills": [1, 0, 0, 0],
        "swamp": [0, 1, 0, 0],
        "forest": [0, 0, 1, 0],
        "plains": [0, 0, 0, 1],
    }

    return terrain2bin.get(terrain, None)


def decode(bin: List[int]) -> str | None:
    bin2terrain = ["hills", "swamp", "forest", "plains"]
    for i, terrain in zip(bin, bin2terrain):
        if i == 1:
            return terrain
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
        for terrain, prototypes in data.items():
            proto = random.choice(prototypes)
            row = blur(proto)
            row[-1] = encode(terrain)
            result.append(row)
    return result


def randrange(low, high):
    return random.uniform(low, high)


def shuffle(data):
    random.shuffle(data)
    return data


def sigmoid(activations):
    return 1.0 / (1.0 + exp(-1.0 * activations))


def feed_forward(features, theta_h, theta_o, n_hidden, n_outputs):
    # activations = [bias, z_0, z_1 ..., z_h] is activation for each hidden node h in n_hidden
    activations = [1.0] + [sigmoid(sum([w_h * x_t for w_h, x_t in zip(theta_h[h], features)])) for h in range(n_hidden)]

    # predictions = [p0, p1, ... pi ] is prediction for each output node i in n_outputs
    predictions = [sigmoid(sum([v_i * z_h for v_i, z_h in zip(theta_o[i], activations)])) for i in range(n_outputs)]
    return activations, predictions


def calculate_error(activations, predictions, label, theta_o, n_outputs, n_hidden):
    delta_o = [predictions[i] * (1 - predictions[i]) * (label[i] - predictions[i]) for i in range(n_outputs)]
    delta_h = [activations[h + 1] * (1 - activations[h + 1]) * sum([theta_o[i][h + 1] * delta_o[i] for i in range(n_outputs)]) for h in range(n_hidden)]
    return delta_h, delta_o


def update_weights(features, activations, theta_h, delta_h, theta_o, delta_o, alpha, n_inputs, n_outputs, n_hidden):
    for i in range(n_outputs):
        for h in range(n_hidden + 1):
            theta_o[i][h] += alpha * delta_o[i] * activations[h]

    for h in range(n_hidden):
        for j in range(n_inputs + 1):
            theta_h[h][j] += alpha * delta_h[h] * features[j]
    return theta_h, theta_o


# n_outputs is number of output nodes
# n_inputs is number of input features per example
# n_hidden is the number of hidden units in hidden layer
def mlp(training, n_outputs, n_inputs, n_hidden, alpha=0.01, epsilon=10**-5, weight_range=0.1, max_iterations=10000, verbose=True, print_freq: int = 1000):
    # weights from input to hidden
    theta_h = [[random.uniform(-weight_range, weight_range) for j in range(n_inputs + 1)] for h in range(n_hidden)]

    # weights from hidden to output
    theta_o = [[random.uniform(-weight_range, weight_range) for h in range(n_hidden + 1)] for i in range(n_outputs)]
    iterations = 0
    prev_error = inf

    while iterations < max_iterations:
        error = 0.0
        for example in training:
            features = [1.0] + example[:-1]  # add bias
            label = example[-1]

            activation, prediction = feed_forward(features, theta_h, theta_o, n_hidden, n_outputs)
            delta_h, delta_o = calculate_error(activation, prediction, label, theta_o, n_outputs, n_hidden)
            theta_h, theta_o = update_weights(features, activation, theta_h, delta_h, theta_o, delta_o, alpha, n_inputs, n_outputs, n_hidden)
            error += sum([(label[i] - prediction[i]) ** 2 for i in range(n_outputs)])

        error /= len(training)
        if error > prev_error:
            alpha /= 10

        if verbose and (iterations % print_freq == 0):
            print(f"{iterations}\t{error:.6f}")

        if abs(error - prev_error) < epsilon:
            return (theta_h, theta_o)

        prev_error = error
        iterations += 1

    return None


def apply_model(model, test_data: List[List[Any]], n_hidden, n_outputs, labeled):
    result = []
    theta_h, theta_o = model
    for example in test_data:
        features = example[:-1] if labeled else example
        features = [1.0] + features
        activations, predictions = feed_forward(features, theta_h, theta_o, n_hidden, n_outputs)

        max_node, max_val = None, -inf
        for i in range(n_outputs):
            y_hat = predictions[i]
            if y_hat > max_val:
                max_val = y_hat
                max_node = i

        row = []
        for i in range(n_outputs):
            prediction = 1 if i == max_node else 0
            if labeled:
                row.append((example[-1][i], prediction))
            else:
                row.append((prediction, predictions[i]))
        result.append(row)
    return result


def evaluate(result):
    miss = 0
    for row in result:
        if (1, 1) not in row:
            miss += 1
    return miss / len(result)

def print_model(model):
    theta_h, theta_o = model

    for weights in theta_h:
        print("|" , end="")
        print(" ".join([str(round(i,3)) for i in weights]), end="")
        print(" |  ", end="")

    print()
    for weights in theta_o:
        print("|" , end="")
        print(" ".join([str(round(i,3)) for i in weights]), end="")
        print(" |  ", end="")
    
    return 


def test2(clean_data):
    dataset = generate_data(clean_data, 100)
    print(len(dataset))
    test_set = generate_data(clean_data, 100000)

    n_inputs = len(dataset[0]) - 1
    n_outputs = len(dataset[0][-1])

    for l in [2, 4, 8]:
        model = mlp(dataset, n_outputs, n_inputs, l)
        assert model is not None
        # print_model(model)

        result = apply_model(model, dataset, l, n_outputs, True)
        error = evaluate(result)
        print("Training Error: ", l, error)
        result = apply_model(model, test_set, l, n_outputs, True)
        error = evaluate(result)
        print("Learning Error: ", l, error)
        print()
    return


test2(clean_data)
