import numpy as np


class FFNetwork:
    def __init__(self, input_dim, output_dim):
        """
        Initializes the classification model with the
        given input and output dimensions.

        :param input_dim: (int) The input dimension of the model.
        :param output_dim: (int) The output dimension of the model.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        np.random.seed(33)
        self.lr = 0.01

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
        return e_x / np.sum(e_x, axis=0, keepdims=True)

    def forward(self, X):
        self.activations[0] = X
        cur = X
        for i in range(len(self.shape) - 2):  # exclude the last
            z = self.weights[i] @ cur + self.biases[i]
            a = self.relu(z)
            self.layer_outputs[i] = z
            self.activations[i + 1] = a
            cur = a
        z_last = self.weights[-1] @ cur + self.biases[-1]
        a_last = self.softmax(z_last)
        self.layer_outputs[-1] = z_last
        self.activations[-1] = a_last
        print(a_last.shape)
        # exit()
        a_last[a_last == 0.0] = 0.001

        loss = -np.mean(np.sum(self.y * np.log(a_last), axis=0))
        print(loss)
        return a_last

    def backward(self, y):
        dz = self.activations[-1] - y
        dw = dz @ self.activations[-2].T
        db = np.mean(dz, axis=1, keepdims=True)
        self.weights[-1] -= self.lr * dw
        self.biases[-1] -= self.lr * db
        for i in range(len(self.shape) - 3, -1, -1):
            da = self.weights[i + 1].T @ dz
            dz = da * self.relu_derivative(self.layer_outputs[i])
            dw = dz @ self.activations[i].T
            db = np.mean(dz, axis=1, keepdims=True)
            self.weights[i] -= self.lr * dw
            self.biases[i] -= self.lr * db

    def one_hot_encode(self, labels):
        if num_classes is None:
            num_classes = np.max(labels) + 1
        one_hot_labels = np.eye(num_classes)[labels]
        return one_hot_labels

    def train(self, X_train, y_train):

        y_train = self.one_hot_encode(y_train)
        print("x shape = ", X_train.shape)
        print("y shape = ", y_train.shape)

        shape = [self.input_dim, 2, self.output_dim]
        self.shape = shape
        np.random.seed(3)
        self.weights = [
            np.random.randn(shape[i + 1], shape[i]) * 0.01
            for i in range(len(shape) - 1)
        ]
        self.biases = [
            np.random.randn(shape[i + 1], 1) * 0.01 for i in range(len(shape) - 1)
        ]
        self.layer_outputs = [None for _ in range(len(shape) - 1)]
        self.activations = [None for _ in range(len(shape))]
        y = y_train.T
        X = X_train.T
        self.y = y
        pretty_print("y", y)
        pretty_print("x", X)
        # exit()

        for _ in range(10):
            self.forward(X)
            self.backward(y)
        exit()

    def predict(self, X_test):
        """
        Makes a prediction on new data using the trained
        classification model.

        :param X_test: (numpy.ndarray) The new data,
        of shape (n_samples, input_dim).
        :return: (numpy.ndarray) The model predictions,
        of shape (n_samples, output_dim).
        """
        X = X_test.T
        pred = self.forward(X)
        labels = np.argmax(pred, axis=0)
        return labels

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the classification model
        on the given test data using classification metrics.

        :param X_test: (numpy.ndarray) The test data,
        of shape (n_samples, input_dim).
        :param y_test: (numpy.ndarray) The test labels,
        of shape (n_samples, output_dim).
        :return: (dict) A dictionary containing the computed
        classification metrics.
        """
        X_test = X_test.T
        y_test = y_test.T

        pred = self.predict(X_test)
        true = y_test.reshape(pred.shape[0])
        mat = np.zeros((self.output_dim, self.output_dim), dtype=int)
        np.add.at(mat, (true, pred), 1)

        true_positive = np.diag(mat)
        false_positive = np.sum(mat, axis=0) - true_positive
        false_negative = np.sum(mat, axis=1) - true_positive

        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        f1_score = 2 * (precision * recall) / (precision + recall)

        # Weighted average
        weights = np.sum(mat, axis=1)
        weighted_precision = np.average(precision, weights=weights)
        weighted_recall = np.average(recall, weights=weights)
        weighted_f1_score = np.average(f1_score, weights=weights)

        return {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1_score": weighted_f1_score,
        }
