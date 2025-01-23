import numpy as np
from decision_tree import DecisionTree


def calculate_entropy(y):
    """
    Calculates the entropy of the labels.

    Parameters:
    - y (numpy.ndarray): Array of labels.

    Returns:
    - float: Entropy value.
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain(X, y, feature_index):
    """
    Computes the information gain for a given feature index.

    Handles both nominal and continuous features.

    Parameters:
    - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
    - y (numpy.ndarray): Labels of shape (n_samples,).
    - feature_index (int): Index of the feature to evaluate.

    Returns:
    - float: Information gain for the feature.
    - callable: Decision criterion for splitting on this feature.
    """
    parent_entropy = calculate_entropy(y)
    feature_values = X[:, feature_index]

    # Check if feature is continuous or nominal
    if np.issubdtype(feature_values.dtype, np.number):
        # Continuous feature: find the best threshold
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_labels = y[sorted_indices]

        best_gain = -1
        best_threshold = None

        # Use cumulative counts to optimize entropy calculation
        total_samples = len(y)
        for i in range(1, len(sorted_values)):
            if sorted_values[i] == sorted_values[i - 1]:
                continue

            left_count = i
            right_count = total_samples - i

            left_labels = sorted_labels[:i]
            right_labels = sorted_labels[i:]

            left_entropy = calculate_entropy(left_labels)
            right_entropy = calculate_entropy(right_labels)

            split_entropy = (
                (left_count / total_samples) * left_entropy
                + (right_count / total_samples) * right_entropy
            )

            gain = parent_entropy - split_entropy

            if gain > best_gain:
                best_gain = gain
                best_threshold = (sorted_values[i] + sorted_values[i - 1]) / 2

        def decision_criterion(sample):
            return sample[feature_index] <= best_threshold

        return best_gain, decision_criterion

    else:
        # Nominal feature: split by unique values
        unique_values = np.unique(feature_values)

        split_entropy = 0
        for value in unique_values:
            subset_indices = feature_values == value
            subset_y = y[subset_indices]
            split_entropy += (len(subset_y) / len(y)) * calculate_entropy(subset_y)

        gain = parent_entropy - split_entropy

        def decision_criterion(sample):
            return sample[feature_index] == unique_values[0]

        return gain, decision_criterion


class ID3(DecisionTree):
    def __init__(self, stopping_criteria=None):
        super().__init__(decision_criteria=self._best_split, stopping_criteria=stopping_criteria)

    def _best_split(self, X, y):
        """
        Finds the best split for the given dataset based on information gain.

        Parameters:
        - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        - y (numpy.ndarray): Labels of shape (n_samples,).

        Returns:
        - dict: Dictionary containing the best decision criterion and impurity decrease.
        """
        best_gain = -1
        best_criterion = None

        for feature_index in range(X.shape[1]):
            gain, criterion = information_gain(X, y, feature_index)
            if gain > best_gain:
                best_gain = gain
                best_criterion = criterion

        return {"impurity_decrease": best_gain, "criterion": best_criterion}

