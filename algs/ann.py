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


# layers ->  Node -> bias+weights
# Model = List[List[List[float]]]
Weights = List[float]
Layer = List[Weights]
Model = Tuple[List[Layer], Layer]


def create_biased_weights(n_weights: int, weight_range: float) -> Weights:
    return [random.uniform(-weight_range, weight_range) for _ in range(n_weights + 1)]


def make_network(n_inputs: int, n_nodes: int, n_layers: int, n_outputs: int, weight_range: float) -> Model:
    hidden_layers = []
    for i in range(n_layers):
        n_weights = n_nodes if i != 0 else n_inputs
        layer = [create_biased_weights(n_weights, weight_range) for _ in range(n_nodes)]
        hidden_layers.append(layer)

    output_layer = [create_biased_weights(n_nodes, weight_range) for _ in range(n_outputs)]
    return (hidden_layers, output_layer)


def feed_forward(model: Model, features: List[float]) -> List[List[float]]:
    activations = []
    biased_input = [1.0] + features
    for layer in model[0] + [model[1]]:
        layer_activations = []
        for nodes in layer:
            z = sum(theta * x for theta, x in zip(nodes, biased_input))
            layer_activations.append(1.0 / (1.0 + exp(-1.0 * z)))
        biased_input = [1.0] + layer_activations
        activations.append(layer_activations)
    return activations


def calculate_output_error(activations: List[List[float]], label: List[float]) -> List[float]:
    deltas = []
    for y_hat, y in zip(activations[-1], label):
        delta = y_hat * (1 - y_hat) * (y - y_hat)
        deltas.append(delta)
    return deltas


def calculate_hidden_error(model: Model, activations: List[List[float]], delta_os: List[float]) -> list[List[float]]:
    deltas = [delta_os]
    n_hidden_layers = len(model[0])
    for i in range(n_hidden_layers):  # calc error from back to front
        layer_idx = n_hidden_layers - i - 1
        layer_deltas = []
        for node_idx in range(len(activations[layer_idx])):
            y_hat = activations[layer_idx][node_idx]
            delta = 0.0
            next_layer = model[0][layer_idx + 1] if layer_idx + 1 < n_hidden_layers else model[1]
            for next_node_idx, next_delta in enumerate(deltas[0]):
                delta += next_layer[next_node_idx][node_idx + 1] * next_delta
            delta *= y_hat * (1 - y_hat)
            layer_deltas.append(delta)
        deltas.insert(0, layer_deltas)
    return deltas


def update_weights(model: Model, features: List[float], activations: List[List[float]], deltas_hs: List[List[float]], alpha: float) -> Model:
    for layer_idx, (layer, layer_deltas) in enumerate(zip(model[0] + [model[1]], deltas_hs)):
        prev_layer_activations = [1.0] + activations[layer_idx - 1] if layer_idx != 0 else [1.0] + features
        for node_idx, (node, delta) in enumerate(zip(layer, layer_deltas)):
            for weight_idx, weight in enumerate(node):
                # weight index is also the node index in the prev layer
                layer[node_idx][weight_idx] = weight + alpha * delta * prev_layer_activations[weight_idx]
    return model


def backprop(model: Model, activations: List[List[float]], features: List[Any], label: List[Any], alpha: float) -> List[List[float]]:
    delta_os = calculate_output_error(activations, label)
    deltas = calculate_hidden_error(model, activations, delta_os)
    model = update_weights(model, features, activations, deltas, alpha)
    return deltas


def learn_model(
    data: List[List[Any]], n_nodes: int, n_layers: int = 1, epsilon: float = 10**-5, alpha: float = 0.01, verbose: bool = True, max_iters: int = 10000, print_freq: int = 1000
) -> Model | None:
    n_inputs = len(data[0]) - 1
    n_outputs = len(data[0][-1])
    model = make_network(n_inputs, n_nodes, n_layers, n_outputs, 0.5)
    iterations = 0
    prev_error = inf
    while iterations < max_iters:
        error = 0.0
        for example in data:
            features, label = example[:-1], example[-1]
            activations = feed_forward(model, features)
            error += sum((y - y_hat) ** 2 for y, y_hat in zip(label, activations[-1]))
            backprop(model, activations, features, label, alpha)
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
    result = []
    for example in test_data:
        features = example[:-1] if labeled else example
        activations = feed_forward(model, features)
        max_node, max_val = None, -inf
        for node, y_hat in enumerate(activations[-1]):
            if y_hat > max_val:
                max_val = y_hat
                max_node = node

        row = []
        for node, (y, y_hat) in enumerate(zip(example[-1], activations[-1])):
            prediction = 1 if node == max_node else 0
            if labeled:
                row.append((y, prediction))
            else:
                row.append((prediction, y_hat))
        result.append(row)
    return result


def evaluate(result):
    miss = 0
    for row in result:
        if (1, 1) not in row:
            miss += 1
    return miss / len(result)


def print_model(model: Model, activations: List[List[float]], deltas: List[List[float]]) -> None:
    print(f"layer,node,weights,activation,delta")
    r = 4
    for layer_idx, (layer, layer_acts, layer_deltas) in enumerate(zip(model[0] + [model[1]], activations, deltas)):
        for node_idx, (weights, g, delta) in enumerate(zip(layer, layer_acts, layer_deltas)):
            print(f"{layer_idx},{node_idx},{[round(w, r) for w in weights]},{round(g, r)},{round(delta, r)}")
    return

def test1():
    input_layer = [0.52, -0.97]
    actual = [1, 0]
    model = ([[[0.01, 0.26, -0.42], [-0.05, 0.78, 0.19], [0.42, -0.23, 0.37]]], [[0.2, 0.61, 0.12, -0.9], [0.3, 0.28, -0.34, 0.10]])
    alpha = 0.01
    activations = feed_forward(model, input_layer)
    delta_hs = backprop(model, activations, input_layer, actual, alpha)
    print_model(model, activations, delta_hs)
    print()
    delta_hs = backprop(model, activations, input_layer, actual, alpha)
    print_model(model, activations, delta_hs)
    return


def test2(clean_data):
    dataset = generate_data(clean_data, 100)
    test_set = generate_data(clean_data, 100)
    for l in [2, 4, 8]:
        model = learn_model(dataset, l)
        assert model is not None
        result = apply_model(model, dataset, True)
        error = evaluate(result)
        print(l, error) 
        result = apply_model(model, test_set, True)
        for r in result:
            actual = decode([i[0] for i in r])
            pred = decode([i[1] for i in r])
            x = "X" if pred != actual else "Y"
            print(x, end="")
        print()
        error = evaluate(result)
        print(l, error)
        print()
    return

# test1()
test2(clean_data)
