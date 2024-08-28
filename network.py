import random
from matrix import Matrix
from rich import print
import json
import time
import math

## https://neuralnetworksanddeeplearning.com/chap2.html#warm_up_a_fast_matrix-based_approach_to_computing_the_output_from_a_neural_network


def index_of_max(lis):
    maxim = float("-inf")
    index = None
    for i, val in enumerate(lis):
        if val > maxim:
            maxim = val
            index = i

    return index


def normalize(inputs):
    total = sum(inputs)
    return [val / total for val in inputs]


def add_vects(x, y):
    return [a + b for a, b in zip(x, y)]


def softmax(input):
    maxim = max(input)
    out = [math.e ** (inp - maxim) for inp in input]
    return normalize(out)


## Activation Functions ##


def ReLU(input):
    return [inp if inp > 0 else 0 for inp in input]


def deriv_ReLU(input):
    return [1 if inp > 0 else 0 for inp in input]


def sigmoid(input, not_list=False):
    if not_list:
        return 1 / (1 + math.e ** (-input))
    return [1 / (1 + math.e ** (-inp)) for inp in input]


def deriv_sigmoid(input, sigmoid_as_input=False):
    if sigmoid_as_input:
        return [inp * (1 - inp) for inp in input]

    return [sigmoid(inp, True) * (1 - sigmoid(inp, True)) for inp in input]


activation_derivative = {
    ReLU: deriv_ReLU,
    sigmoid: deriv_sigmoid,
}

activations = {
    "ReLU": ReLU,
    "sigmoid": sigmoid,
}


class Neural_Net:
    def __init__(self, *layers):
        self.layers = []
        for i, lay in enumerate(layers):
            if i == 0:
                prev = lay
                continue
            self.layers.append(Layer(prev, lay))
            prev = lay

        self.lr = 0.1
        self.activation = "ReLU"
        self.act_func = ReLU

    def set_learning_rate(self, lr):
        self.lr = lr

    def set_activation(self, func):
        self.activation = func
        self.act_func = activations[func]

    def feed_forward(self, inputs):
        out = inputs
        for lay in self.layers[:-1]:
            out = lay.propogate(out, self.act_func)

        # output layer is applied to softmax
        out = self.layers[-1].propogate(out)
        # softmax shouldn't be applied if only 1 output node
        return softmax(out)
        # return self.layers[-1].propogate(out)

    def cost_gradient(self, output, targets):
        if not len(output) == len(targets):
            raise Exception("Invalid targets given")
        return [[exp - out] for out, exp in zip(output, targets)]

    def backpropogation_alg(self, error: Matrix):
        output_layer = True

        for layer in reversed(self.layers):
            if output_layer:
                prev_weights = layer.weights
                layer.change(error, self.lr)
                output_layer = False
            else:
                # hidden_error = (w^(l+1)_transpose * error^(l+1)) (hadamard) derivative_activation(z^(l))
                error = prev_weights.transpose() * error
                act_prime = Matrix(
                    1,
                    len(layer.output),
                    [activation_derivative[self.act_func](layer.output)],
                )
                error = error.hadamard(act_prime.transpose())
                prev_weights = layer.weights
                layer.change(error, self.lr)

    def backpropogate(self, inputs, targets):
        # output_error = cost_gradient (hadamard) derivative_activation(z)
        output = self.feed_forward(inputs)
        error = self.cost_gradient(output, targets)
        error = Matrix(len(error), 1, error)

        self.backpropogation_alg(error)

    def backpropogate_w_batches(self, inputs, targets, batch_size):
        error = Matrix(len(targets[0]), 1)
        for input, target in zip(inputs, targets):
            output = self.feed_forward(input)
            error = error + Matrix(len(output), 1, self.cost_gradient(output, target))

        error * (1 / batch_size)
        # error = Matrix(len(error), 1, error)

        self.backpropogation_alg(error)

    def calculate_vals(self, data):
        length = len(data)
        vals = [length / 10 * y for y in range(10)]
        vals = set(vals)
        print(vals)
        return vals, length

    def progress_update(self, start_time, index, markers, data_length):
        if index in markers:
            end = time.time()
            print(
                f"[red]Progress Update: {index/data_length*100}, Time: {end - start_time}"
            )
            return end
        return start_time

    def train(self, data):
        # assumes data is already shuffled when given
        vals, length = self.calculate_vals(data)

        start = time.time()
        for i, d in enumerate(data):
            start = self.progress_update(start, i, vals, length)
            self.backpropogate(d["input"], d["output"])

    def train_w_batches(self, data, batch_size):
        vals, length = self.calculate_vals(data)

        start = time.time()
        for b in range(0, length, batch_size):
            start = self.progress_update(start, b, vals, length)

            inputs, targets = [], []
            for d in data[b : b + batch_size]:
                inputs.append(d["input"])
                targets.append(d["output"])

            self.backpropogate_w_batches(inputs, targets, batch_size)

    def evaluate(self, test):
        correct = 0
        for t in test:
            guess = self.feed_forward(t["input"])
            ind = index_of_max(guess)
            if t["output"][ind] == 1:
                correct += 1

        return correct / len(test)

    def serialize(self):
        dict_representation = {
            "learning_rate": self.lr,
            "activation": self.activation,
        }

        for i, layer in enumerate(self.layers):
            dict_representation[f"layer_{i}"] = layer.__dict__()

        return dict_representation

    def deserialize(self, encoding: dict):
        self.set_learning_rate(encoding["learning_rate"])
        self.set_activation(encoding["activation"])
        self.layers = []

        for key, value in encoding.items():
            if key in {"learning_rate", "activation"}:
                continue

            wt_info = value["weights"]
            wt_matrix = Matrix(wt_info["rows"], wt_info["cols"], wt_info["vals"])

            self.layers.append(Layer(weights=wt_matrix, biases=value["biases"]))

    def save(self, filename="nn_model.json"):
        """
        Saves model, including layers, weights, and biases, as a json file
        """
        with open(filename, "w") as json_file:
            json.dump(self.serialize(), json_file, indent=4)

    def load(self, filename):
        """
        changes self into a pre-saved model
        """
        with open(filename, "r") as json_file:
            brain = json.load(json_file)

        self.deserialize(brain)


class Layer:
    def __init__(self, inputs=None, nodes=None, weights=None, biases=None):
        if not weights is None:
            self.weights = weights
        else:
            self.weights = Matrix(nodes, inputs, randomizer=True)

        if not biases is None:
            self.biases = biases
        else:
            self.biases = [random.uniform(-1, 1) for _ in range(nodes)]

        self.input_recieved = None
        self.output = None
        self.activation_result = None

    def propogate(self, inputs, func=None):
        self.input_recieved = inputs
        self.output = add_vects(self.weights * inputs, self.biases)
        if func is None:
            self.activation_result = self.output
            return self.output

        self.activation_result = func(self.output)

        return self.activation_result

    def change(self, error: Matrix, lr):
        # dW = dZ(l) * output_transpose^(l-1)
        # dB = dZ(l)
        delta_W = error * Matrix(1, len(self.input_recieved), [self.input_recieved])
        delta_B = error
        delta_W * lr
        delta_B * lr

        self.weights += delta_W
        self.biases = add_vects(self.biases, delta_B.converter())

    def __dict__(self):
        return {
            "weights": {
                "rows": self.weights.rows,
                "cols": self.weights.cols,
                "vals": self.weights.vals,
            },
            "biases": self.biases,
        }

    def __str__(self):
        return f"Layer({self.weights.cols}, {self.weights.rows}), \nWeights: {self.weights}"


if __name__ == "__main__":
    # nn = Neural_Net(2, 10, 1)

    # data = [
    #     {"input": (1, 1), "output": (0,)},
    #     {"input": (0, 0), "output": (0,)},
    #     {"input": (1, 0), "output": (1,)},
    #     {"input": (0, 1), "output": (1,)},
    #     # {"input": (0.1, 0.1), "output": (0.1,)},
    #     # {"input": (0.3, 0.7), "output": (0.8,)}
    # ]

    # nn.set_activation(ReLU)
    # nn.train_w_shuffle(data, 5000)

    # print("[blue]Test")
    # for d in data:
    #     result = np.round(nn.feed_forward(d["input"]), 2)
    #     if result[0] < d["output"][0] + 0.05 and result[0] > d["output"][0] - 0.05:
    #         print(f"[green]{d["input"]}: {result[0]}")
    #     else:
    #         print(f"[red]{d["input"]}: {result[0]}")

    pass
