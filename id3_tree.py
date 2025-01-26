import numpy as np
from decision_tree import DecisionTree
import pandas as pd


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
    # === Missing data skip for continuous ===
    # Filter out missing values (np.nan) along with their labels
    not_missing_mask = ~pd.isna(X_col)
    X_col_clean = X_col[not_missing_mask]
    y_clean = y[not_missing_mask]

    # Edge case: if everything is missing, no valid split
    if len(X_col_clean) == 0:
        return 0.0, None

    # Sort by feature values
    sorted_indices = np.argsort(X_col_clean)
    sorted_values = X_col_clean[sorted_indices]
    sorted_labels = y_clean[sorted_indices]

    best_gain = -1.0
    best_threshold = None
    total_samples = len(sorted_labels)

    # We will skip repeated values to avoid redundant splits
    for i in range(1, total_samples):
        if sorted_values[i] == sorted_values[i - 1]:
            continue

        threshold = (sorted_values[i] + sorted_values[i - 1]) / 2.0

        left_labels = sorted_labels[:i]
        right_labels = sorted_labels[i:]

        left_entropy = calculate_entropy(left_labels)
        right_entropy = calculate_entropy(right_labels)

        split_entropy = (
            (len(left_labels) / total_samples) * left_entropy
            + (len(right_labels) / total_samples) * right_entropy
        )
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

    For missing data, we skip them during gain computation.
    """

    # === Missing data skip for nominal ===
    # We'll treat None as missing for nominal; skip those
    not_missing_mask = ~pd.isna(X_col)
    X_col_clean = X_col[not_missing_mask]
    y_clean = y[not_missing_mask]
    print(len(X_col_clean))
    if len(X_col_clean) == 0:
        # If all are missing for this feature, no valid split
        return 0.0, None

    unique_values = np.unique(X_col_clean)

    # If there's only one unique non-missing category, no real split
    if len(unique_values) <= 1:
        return 0.0, unique_values[0] if len(unique_values) == 1 else None

    best_gain = -1.0
    best_value = None

    for val in unique_values:
        left_indices = (X_col_clean == val)
        right_indices = (X_col_clean != val)

        left_entropy = calculate_entropy(y_clean[left_indices]) if np.any(left_indices) else 0
        right_entropy = calculate_entropy(y_clean[right_indices]) if np.any(right_indices) else 0

        left_ratio = np.sum(left_indices) / len(y_clean)
        right_ratio = np.sum(right_indices) / len(y_clean)

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
                 min_impurity_decrease=0.0, continuous_features_indexes=None):
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
        if continuous_features_indexes is None:
            continuous_features_indexes = []
        self.continuous_features_indexes = continuous_features_indexes

    def _best_split(self, X, y):
        """
        Finds the best split for the given dataset based on Information Gain,
        penalizing features with high missing rate.

        Returns
        -------
        dict
            {
                "impurity_decrease": float,  # Best 'penalized' information gain
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

            # Identify missing vs. non-missing
            not_missing_mask = ~np.isnan(feature_values) if np.issubdtype(feature_values.dtype, np.number) \
                                 else np.array([val is not None for val in feature_values])

            fraction_nonmissing = np.sum(not_missing_mask) / n_samples

            # Calculate gain only on non-missing subset
            if feature_index in self.continuous_features_indexes:
                gain, threshold = information_gain_continuous(feature_values, y, parent_entropy)
                if threshold is not None:
                    # Apply penalty: multiply gain by fraction_nonmissing
                    penalized_gain = gain * fraction_nonmissing
                    if penalized_gain > best_gain:
                        best_gain = penalized_gain
                        best_feature = feature_index
                        best_threshold = threshold
                        best_criterion = lambda row, thr=threshold, f=feature_index: (
                            not self._is_missing(row[f]) and (row[f] <= thr)
                        )
            else:
                gain, chosen_val = information_gain_nominal(feature_values, y, parent_entropy)
                if chosen_val is not None:
                    # Apply penalty
                    penalized_gain = gain * fraction_nonmissing
                    if penalized_gain > best_gain:
                        best_gain = penalized_gain
                        best_feature = feature_index
                        best_threshold = chosen_val
                        best_criterion = lambda row, val=chosen_val, f=feature_index: (
                            (not self._is_missing(row[f])) and (row[f] == val)
                        )

        return {
            "impurity_decrease": best_gain if best_gain > 0 else 0.0,
            "criterion": best_criterion,
            "feature_idx": best_feature,
            "threshold": best_threshold
        }
