import numpy as np


class DecisionTree:
    def __init__(self, max_depth=None, prev_pred=None):
        self.max_depth = max_depth
        self.feature_count = None

    def _calculate_entropy(self, y):
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            entropy -= p_cls * np.log2(p_cls)
        return entropy

    def _split(self, X, feature_index, threshold):
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return left_mask, right_mask

    def _find_best_split(self, X_train, y_train):
        rows = X_train.shape[0]
        random_mask = np.random.choice(rows, int(rows / 4), replace=False)
        X = X_train[random_mask]
        y = y_train[random_mask]

        n_features = X.shape[1]
        best_crit = float("inf")
        best_feature_index = None
        best_threshold = None

        feature_index = np.random.choice(n_features)

        # check for continuous or discrete faeature
        if np.all(np.mod(X[:, feature_index], 1) == 0):
            thresholds = np.unique(X[:, feature_index])
        else:
            thresholds = np.linspace(
                X[:, feature_index].min(), X[:, feature_index].max(), 10
            )

        for threshold in thresholds:
            left_mask, right_mask = self._split(X, feature_index, threshold)
            y_left = y[left_mask]
            y_right = y[right_mask]
            if len(y_left) < 2 or len(y_right) < 2:
                continue
            crit_left = self._calculate_entropy(y_left)
            crit_right = self._calculate_entropy(y_right)
            crit = (len(y_left) * crit_left + len(y_right) * crit_right) / len(y)
            if crit < best_crit:
                best_crit = crit
                best_feature_index = feature_index
                best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, X_train, y_train, depth):
        if depth == self.max_depth:
            return np.mean(y_train, axis=0)

        best_feature_index, best_threshold = self._find_best_split(X_train, y_train)

        if best_feature_index is None:
            return np.mean(y_train, axis=0)

        # print("index , threshold = ", best_feature_index, best_threshold)
        left_mask, right_mask = self._split(X_train, best_feature_index, best_threshold)
        node = {}
        node["feature_index"] = best_feature_index
        node["threshold"] = best_threshold
        node["size"] = len(y_train)
        node["left"] = self._build_tree(
            X_train[left_mask], y_train[left_mask], depth + 1
        )
        node["right"] = self._build_tree(
            X_train[right_mask], y_train[right_mask], depth + 1
        )

        return node

    def fit(self, X_train, y_train, feature_count=None):
        self.feature_count = feature_count
        self.tree_ = self._build_tree(X_train, y_train, 0)

    def _predict_instance(self, x, tree):
        if isinstance(tree, dict):
            feature_index = tree["feature_index"]
            if x[feature_index] <= tree["threshold"]:
                return self._predict_instance(x, tree["left"])
            else:
                return self._predict_instance(x, tree["right"])
        else:
            return tree

    def predict(self, X):
        predictions = []
        for x in X:
            predictions.append(self._predict_instance(x, self.tree_))
        return np.array(predictions)


class GradientBoosting:
    def __init__(self, input_dim, output_dim):
        """
        Initializes the classification model with the
        given input and output dimensions.

        :param input_dim: (int) The input dimension of the model.
        :param output_dim: (int) The output dimension of the model.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.base = None
        self.dtrees = []
        self.base_val = None
        self._learning_rate = 0.025

    def calculate_base_predition(self, y_train):
        return 1 / self.output_dim

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def one_hot_encode(self, labels):
        num_samples = len(labels)
        encoded_labels = np.zeros((num_samples, self.output_dim))
        for i in range(num_samples):
            class_index = int(labels[i])
            encoded_labels[i, class_index] = 1
        return encoded_labels

    def fit_transform(self, X):
        self.means = np.mean(X, axis=0)
        self.stds = np.std(X, axis=0)
        return (X - self.means) / self.stds

    def transform(self, X):
        return (X - self.means) / self.stds

    def train(self, X_train, y_train):
        """
        Trains the classification model on the given training data.

        :param X_train: (numpy.ndarray) The training data,
        of shape (n_samples, input_dim).
        :param y_train: (numpy.ndarray) The training labels,
        of shape (n_samples, output_dim).
        """
        X_train = self.fit_transform(X_train)
        y_train = self.one_hot_encode(y_train)
        self.base_val = self.calculate_base_predition(
            y_train
        )  # allows us to use the same value for predict
        self.base = np.full((X_train.shape[0], self.output_dim), self.base_val)
        current_pred = self.base
        for _ in range(250):
            X_sample, y_sample = X_train, y_train
            X_sample, y_sample = self._bootstrap_sample(X_train, y_train)
            residual = y_sample - current_pred
            tree = DecisionTree(max_depth=8)
            tree.fit(X_sample, residual)
            tree_pred = tree.predict(X_train)
            if tree_pred.ndim == 1:
                tree_pred = tree_pred[:, np.newaxis]
            current_pred += self._learning_rate * tree_pred
            self.dtrees.append(tree)
            # print()

    def predict(self, X_test):
        """
        Makes a prediction on new data using the trained
        classification model.

        :param X_test: (numpy.ndarray) The new data,
        of shape (n_samples, input_dim).
        :return: (numpy.ndarray) The model predictions,
        of shape (n_samples, output_dim).
        """

        X_test = self.transform(X_test)
        pred = np.full((X_test.shape[0], self.output_dim), self.base_val)
        for d in self.dtrees:
            pred += self._learning_rate * d.predict(X_test)
        labels = np.argmax(pred, axis=1)
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
