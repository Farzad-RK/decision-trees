import numpy as np
from decision_tree import DecisionTree  # Your refactored scikit-learn-compatible base


# Make sure the import path is correct for your setup.


def calculate_entropy(y):
    """
    Calculate the entropy of the labels array `y`.

    Entropy = -sum(p_i * log2(p_i)), over all classes i

    Parameters
    ----------
    y : np.ndarray
        Array of labels of shape (n_samples,).

    Returns
    -------
    float
        Entropy value.
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))


def information_gain_continuous(X_col, y, parent_entropy):
    """
    Compute the maximum information gain for a continuous feature
    and the best threshold.

    Parameters
    ----------
    X_col : np.ndarray
        The 1D array of feature values (continuous).
    y : np.ndarray
        The labels for each sample.
    parent_entropy : float
        Entropy of the parent node.

    Returns
    -------
    best_gain : float
        The maximum information gain found.
    best_threshold : float
        The threshold that yields the best information gain.
    """
    # Sort by feature values
    sorted_indices = np.argsort(X_col)
    sorted_values = X_col[sorted_indices]
    sorted_labels = y[sorted_indices]

    best_gain = -1.0
    best_threshold = None
    total_samples = len(y)

    # We will skip repeated values to avoid redundant splits
    for i in range(1, total_samples):
        # If the feature value is the same as previous, skip
        if sorted_values[i] == sorted_values[i - 1]:
            continue

        # Split at midpoint between distinct consecutive values
        threshold = (sorted_values[i] + sorted_values[i - 1]) / 2.0

        # Left side: up to i-1
        left_labels = sorted_labels[:i]
        right_labels = sorted_labels[i:]

        left_entropy = calculate_entropy(left_labels)
        right_entropy = calculate_entropy(right_labels)

        # Weighted average entropy after split
        split_entropy = (
                (len(left_labels) / total_samples) * left_entropy
                + (len(right_labels) / total_samples) * right_entropy
        )
        # Information Gain
        gain = parent_entropy - split_entropy

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold


def information_gain_nominal(X_col, y, parent_entropy):
    """
    Compute the information gain for a nominal/categorical feature.

    Since the base DecisionTree uses binary splits (left vs right),
    we will do a 2-way split:
        - Left branch: X_col == first_unique_value
        - Right branch: everything else

    Parameters
    ----------
    X_col : np.ndarray
        The 1D array of feature values (nominal).
    y : np.ndarray
        The labels for each sample.
    parent_entropy : float
        Entropy of the parent node.

    Returns
    -------
    gain : float
        The information gain from this binary partition.
    first_value : object
        The value that goes to the "left" branch.
    """
    unique_values = np.unique(X_col)
    # Simple approach: pick one value to go "left", everything else "right"
    # For a basic ID3 approach with multi-way splits, we'd branch on each value
    # but this base class is strictly binary, so we do a 2-part partition.

    if len(unique_values) == 1:
        # There's only one category, so no split actually occurs
        return 0.0, unique_values[0]

    # As a simple heuristic, just pick the most frequent category
    # to isolate on the "left" side, to see if that yields a good gain.
    # Alternatively, we could try each unique value in turn and pick the best.
    best_gain = -1.0
    best_value = None

    for val in unique_values:
        left_indices = (X_col == val)
        right_indices = (X_col != val)

        left_entropy = calculate_entropy(y[left_indices]) if np.any(left_indices) else 0
        right_entropy = calculate_entropy(y[right_indices]) if np.any(right_indices) else 0

        left_ratio = np.sum(left_indices) / len(y)
        right_ratio = np.sum(right_indices) / len(y)

        split_entropy = (left_ratio * left_entropy) + (right_ratio * right_entropy)
        gain = parent_entropy - split_entropy

        if gain > best_gain:
            best_gain = gain
            best_value = val

    return best_gain, best_value


class ID3(DecisionTree):
    """
    ID3 algorithm using Information Gain (based on Entropy).

    In classical ID3:
      - We do multi-way splits for nominal features.
      - Here, we do a binary split (True/False branches) to match
        the base DecisionTree's left_child/right_child structure.
      - For continuous features, we find the best threshold.
    """

    def __init__(self, stopping_criteria=None, max_depth=None, min_samples_split=2,
                 min_impurity_decrease=0.0):
        """
        ID3 Constructor. Inherits from the scikit-learn-compatible DecisionTree base class.

        Parameters
        ----------
        stopping_criteria : dict, optional
            Additional stopping criteria, e.g. {"max_depth": 5}.
        max_depth : int, optional
            Max tree depth to allow.
        min_samples_split : int, optional
            Minimum samples required to split a node.
        min_impurity_decrease : float, optional
            Minimum info gain needed to allow a split.
        """
        super().__init__(
            decision_criteria=self._best_split,
            stopping_criteria=stopping_criteria,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease
        )

    def _best_split(self, X, y):
        """
        Finds the best split for the given dataset based on Information Gain.

        Returns
        -------
        dict
            {
                "impurity_decrease": float,  # Best information gain
                "criterion": callable,       # A function x -> True/False for the best split
                "feature_idx": int,          # Index of the best feature
                "threshold": float or object # For continuous: threshold
                                             # For nominal: the chosen category
            }
        """
        parent_entropy = calculate_entropy(y)
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        best_criterion = None

        n_samples, n_features = X.shape

        # Evaluate each feature
        for feature_index in range(n_features):
            feature_values = X[:, feature_index]

            if np.issubdtype(feature_values.dtype, np.number):
                # Continuous feature
                gain, threshold = information_gain_continuous(feature_values, y, parent_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = threshold
                    # The criterion for a sample: is feature <= threshold?
                    best_criterion = lambda row, thr=threshold, f=feature_index: row[f] <= thr
            else:
                # Nominal (categorical) feature
                gain, chosen_val = information_gain_nominal(feature_values, y, parent_entropy)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_index
                    best_threshold = chosen_val
                    # The criterion for a sample: is feature == chosen_val?
                    best_criterion = lambda row, val=chosen_val, f=feature_index: row[f] == val

        # Return dictionary in the format expected by the base DecisionTree
        return {
            "impurity_decrease": best_gain,
            "criterion": best_criterion,
            "feature_idx": best_feature,
            "threshold": best_threshold
        }
