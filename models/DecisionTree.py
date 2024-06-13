import numpy as np


class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        # Initialize the decision tree with specified hyperparameters.
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _calculate_gini(self, y):
        # Calculate Gini impurity for a given set of labels.
        classes = np.unique(y)
        gini = 1
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            gini -= p_cls**2
        return gini

    def _calculate_misclassification_error(self, y):
        classes = np.unique(y)
        error = 1 - np.max([np.sum(y == cls) / len(y) for cls in classes])
        return error

    def _calculate_entropy(self, y):
        # Calculate entropy for a given set of labels.
        classes = np.unique(y)
        entropy = 0
        for cls in classes:
            p_cls = np.sum(y == cls) / len(y)
            entropy -= p_cls * np.log2(p_cls)
        return entropy

    def _calculate_criterion(self, y):
        # Calculate impurity/criterion based on the specified criterion.
        if self.criterion == "gini":
            return self._calculate_gini(y)
        elif self.criterion == "entropy":
            return self._calculate_entropy(y)
        elif self.criterion == "mis_class":
            return self._calculate_misclassification_error(y)
        else:
            raise ValueError("Invalid criterion. Use 'gini' or 'entropy'.")

    def _split(self, X, y, feature_index, threshold):
        # Split dataset based on a feature and a threshold.
        left_mask = X[:, feature_index] <= threshold
        right_mask = ~left_mask
        return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

    def _find_best_split(self, X, y):
        # Find the best feature and threshold to split the dataset.
        n_features = X.shape[1]
        best_gini = float("inf")
        best_feature_index = None
        best_threshold = None

        for feature_index in range(n_features):

            # Check if the feature is continuous or discrete
            if np.all(np.mod(X[:, feature_index], 1) == 0):
                thresholds = np.unique(X[:, feature_index])
            else:
                # If the feature values are not all integers (continuous feature),
                # generate 10 equally spaced thresholds between the minimum and maximum feature values.
                thresholds = np.linspace(
                    X[:, feature_index].min(), X[:, feature_index].max(), 15
                )

            for threshold in thresholds:
                _, _, y_left, y_right = self._split(X, y, feature_index, threshold)
                if (
                    len(y_left) < self.min_samples_split
                    or len(y_right) < self.min_samples_split
                ):
                    continue

                gini_left = self._calculate_criterion(y_left)
                gini_right = self._calculate_criterion(y_right)
                gini = (len(y_left) * gini_left + len(y_right) * gini_right) / len(y)
                if gini < best_gini:
                    best_gini = gini
                    best_feature_index = feature_index
                    best_threshold = threshold

        return best_feature_index, best_threshold

    def _build_tree(self, X, y, depth):
        # Recursively build the decision tree.
        if depth == self.max_depth or len(np.unique(y)) == 1:
            return Counter(y).most_common(1)[0][0]

        best_feature_index, best_threshold = self._find_best_split(X, y)
        if best_feature_index is None:
            return Counter(y).most_common(1)[0][0]

        X_left, X_right, y_left, y_right = self._split(
            X, y, best_feature_index, best_threshold
        )

        node = {}
        node["feature_index"] = best_feature_index
        node["threshold"] = best_threshold
        node["size"] = len(y)
        node["left"] = self._build_tree(X_left, y_left, depth + 1)
        node["right"] = self._build_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X, y):
        # Fit the decision tree to the training data.
        self.tree_ = self._build_tree(X, y, 0)

    def _predict_instance(self, x, tree):
        # Predict the label of a single instance.
        if isinstance(tree, dict):
            feature_index = tree["feature_index"]
            if x[feature_index] <= tree["threshold"]:
                return self._predict_instance(x, tree["left"])
            else:
                return self._predict_instance(x, tree["right"])
        else:
            return tree

    def predict(self, X):
        # Predict labels for multiple instances.
        predictions = []
        for x in X:
            predictions.append(self._predict_instance(x, self.tree_))
        return np.array(predictions)

    def print_tree(self):
        # Print the decision tree in a readable format.
        self._print_node(self.tree_, 0)

    def _print_node(self, node, depth):
        # Recursively print nodes of the decision tree.
        if isinstance(node, dict):
            print(
                "  " * depth,
                f"Feature: {node['feature_index']}, Threshold: {node['threshold']} , Size = {node['size']}",
            )
            self._print_node(node["left"], depth + 1)
            self._print_node(node["right"], depth + 1)
        else:
            print("  " * depth, f"Leaf node: Class {node}")

    def plot_tree_text(self, node=None, depth=0):
        # Plot the decision tree in text format.
        if node is None:
            node = self.tree_

        if isinstance(node, dict):
            print(
                f"{' ' * depth * 2} [X{node['feature_index']} <= {node['threshold']} , Size = {node['size']}"
            )
            self.plot_tree_text(node["left"], depth + 1)
            self.plot_tree_text(node["right"], depth + 1)
        else:
            print(f"{' ' * depth * 2} [{node}]")
