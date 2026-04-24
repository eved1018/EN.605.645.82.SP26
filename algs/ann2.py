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


def randrange(low, high):
    return random.uniform(low, high)


def shuffle(data):
    random.shuffle(data)
    return data


def sigmoid(z):
    return 1.0 / (1.0 + exp(-1.0 * z))


def feed_forward(x, w, v, H, K):
    if len(x) != len(w[0]):
        return None

    # z = [bias, z_0, z_1 ..., z_h] is activation for each hidden node h in H
    z = [1.0] + [sigmoid(sum([w_h * x_t for w_h, x_t in zip(w[h], x)])) for h in range(H)]

    if len(z) != len(v[0]):
        return None

    # y = [p0, p1, ... pi ] is prediction for each output node i in K
    y = [sigmoid(sum([v_i * z_h for v_i, z_h in zip(v[i], z)])) for i in range(K)]
    return y, z


# K is number of output nodes
# d is number of input features per example
# H is the number of hidden units in hidden layer
def mlp(training, K, d, H, alpha=0.01, epsilon=10 * -5, weight_range=0.01, max_iterations=10000):
    w = [[randrange(-weight_range, weight_range) for j in range(d + 1)] for h in range(H)]  # weights from input to hidden
    v = [[randrange(-weight_range, weight_range) for h in range(H + 1)] for i in range(K)]  # weights from hidden to output

    iterations = 0
    prev_error = inf

    while iterations < max_iterations:
        error = 0.0
        for t in shuffle(training):
            x = [1.0] + t[:-1]  # add bias
            r = t[-1]

            a = feed_forward(x, w, v, H, K)
            if a is None:
                return None

            y, z = a

            dv, dw = [], []
            for i in range(K):
                dv.append([alpha * (r[i] - y[i]) * z[h] for h in range(H)])

            for h in range(H):
                s = sum([(r[i] - y[i]) * v[i][h] for i in range(K)])
                c = alpha * s * z[h] * (1 - z[h])
                dw.append([alpha * c * x[j] for j in range(d)])

            for i in range(K):
                for h in range(H):
                    v[i][h] = v[i][h] + dv[i][h]

            for h in range(H):
                for j in range(d):
                    w[h][j] = w[h][j] + dw[h][j]
            error += sum([(r[i] - y[i]) ** 2 for i in range(K)])

        error /= len(training)
        if error > prev_error:
            alpha /= 10

        if abs(error - prev_error) < epsilon:
            return (w, v)

    return None


def apply_model(model, test_data: List[List[Any]], H, labeled=False):
    result = []
    w, v = model
    n_outputs = len(test_data[0][-1])
    for example in test_data:
        features = example[:-1] if labeled else example
        activations, _ = feed_forward(features, w, v, H, n_outputs)
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


def test2(clean_data):
    dataset = generate_data(clean_data, 100)
    test_set = generate_data(clean_data, 100)
    d = len(test_set[0]) - 1

    n_inputs = len(dataset[0]) - 1
    n_outputs = len(dataset[0][-1])

    for l in [2, 4, 8]:
        model = mlp(dataset, n_outputs, n_inputs, l)
        # model = learn_model(dataset, l)
        assert model is not None
        result = apply_model(model, dataset, l, True)
        error = evaluate(result)
        print(l, error)
        result = apply_model(model, test_set, l, True)
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


test2(clean_data)
