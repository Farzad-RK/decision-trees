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
    return -np.sum(probabilities * np.log2(probabilities + 1e-12))  # +1e-12 to avoid log(0)


def calculate_gini(y):
    """
    Calculate the Gini impurity of the labels array `y`.

    Gini = 1 - sum(p_i^2), over all classes i

    Parameters
    ----------
    y : np.ndarray
        Array of labels of shape (n_samples,).

    Returns
    -------
    float
        Gini impurity value.
    """
    classes, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1.0 - np.sum(probabilities**2)


def information_gain_continuous(X_col, y, parent_entropy):
    """
    Compute the maximum information gain for a continuous feature
    and the best threshold. (For criterion='information_gain' only)

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
    not_missing_mask = ~pd.isna(X_col)
    X_col_clean = X_col[not_missing_mask]
    y_clean = y[not_missing_mask]

    if len(X_col_clean) == 0:
        return 0.0, None

    # Sort by feature values
    sorted_indices = np.argsort(X_col_clean)
    sorted_values = X_col_clean[sorted_indices]
    sorted_labels = y_clean[sorted_indices]

    best_gain = -1.0
    best_threshold = None
    total_samples = len(sorted_labels)

    for i in range(1, total_samples):
        if sorted_values[i] == sorted_values[i - 1]:
            continue

        threshold = (sorted_values[i] + sorted_values[i - 1]) / 2.0

        left_labels = sorted_labels[:i]
        right_labels = sorted_labels[i:]

        left_entropy = calculate_entropy(left_labels)
        right_entropy = calculate_entropy(right_labels)
        split_entropy = (len(left_labels)/total_samples)*left_entropy \
                      + (len(right_labels)/total_samples)*right_entropy
        gain = parent_entropy - split_entropy

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold


def gain_ratio_continuous(X_col, y, parent_entropy):
    """
    Compute the maximum gain ratio for a continuous feature and the best threshold.
    (For criterion='gain_ratio').

    Gain Ratio = Information Gain / SplitInfo
    where SplitInfo = -sum_{branches}(p_i * log2(p_i)).

    Returns
    -------
    best_ratio : float
        The maximum gain ratio found.
    best_threshold : float
        The threshold that yields the best gain ratio.
    """
    not_missing_mask = ~pd.isna(X_col)
    X_col_clean = X_col[not_missing_mask]
    y_clean = y[not_missing_mask]

    if len(X_col_clean) == 0:
        return 0.0, None

    sorted_indices = np.argsort(X_col_clean)
    sorted_values = X_col_clean[sorted_indices]
    sorted_labels = y_clean[sorted_indices]

    best_ratio = -1.0
    best_threshold = None
    total_samples = len(sorted_labels)

    parent_enth = calculate_entropy(y_clean)

    for i in range(1, total_samples):
        if sorted_values[i] == sorted_values[i - 1]:
            continue

        threshold = (sorted_values[i] + sorted_values[i - 1]) / 2.0

        left_labels = sorted_labels[:i]
        right_labels = sorted_labels[i:]

        left_entropy = calculate_entropy(left_labels)
        right_entropy = calculate_entropy(right_labels)
        p_left = len(left_labels) / total_samples
        p_right = len(right_labels) / total_samples

        # Information gain
        child_entropy = p_left * left_entropy + p_right * right_entropy
        ig = parent_enth - child_entropy

        # Split info
        split_info = 0.0
        if p_left > 0:
            split_info -= p_left * np.log2(p_left)
        if p_right > 0:
            split_info -= p_right * np.log2(p_right)

        # Gain ratio
        if split_info > 0:
            ratio = ig / split_info
        else:
            ratio = 0.0

        if ratio > best_ratio:
            best_ratio = ratio
            best_threshold = threshold

    return best_ratio, best_threshold

def gini_continuous(X_col, y, parent_gini):
    """
    Compute the maximum Gini decrease for a continuous feature
    and the best threshold. (For criterion='gini').

    Gini decrease = parent_gini - [weighted average of children gini].
    """
    not_missing_mask = ~pd.isna(X_col)
    X_col_clean = X_col[not_missing_mask]
    y_clean = y[not_missing_mask]

    if len(X_col_clean) == 0:
        return 0.0, None

    sorted_indices = np.argsort(X_col_clean)
    sorted_values = X_col_clean[sorted_indices]
    sorted_labels = y_clean[sorted_indices]

    best_gain = -1.0
    best_threshold = None
    total_samples = len(sorted_labels)

    for i in range(1, total_samples):
        if sorted_values[i] == sorted_values[i - 1]:
            continue

        threshold = (sorted_values[i] + sorted_values[i - 1]) / 2.0

        left_labels = sorted_labels[:i]
        right_labels = sorted_labels[i:]

        left_gini = calculate_gini(left_labels)
        right_gini = calculate_gini(right_labels)
        child_gini = (len(left_labels)/total_samples)*left_gini \
                   + (len(right_labels)/total_samples)*right_gini
        gain = parent_gini - child_gini

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_gain, best_threshold



def information_gain_nominal(X_col, y, parent_entropy):
    """
    Compute the information gain for a nominal/categorical feature (binary split).
    For criterion='information_gain' only.
    """
    not_missing_mask = ~pd.isna(X_col)
    X_col_clean = X_col[not_missing_mask]
    y_clean = y[not_missing_mask]

    if len(X_col_clean) == 0:
        return 0.0, None

    unique_values = np.unique(X_col_clean)
    if len(unique_values) <= 1:
        return 0.0, unique_values[0] if len(unique_values) == 1 else None

    best_gain = -1.0
    best_value = None

    for val in unique_values:
        left_indices = (X_col_clean == val)
        right_indices = (X_col_clean != val)

        left_entropy = calculate_entropy(y_clean[left_indices]) if np.any(left_indices) else 0
        right_entropy = calculate_entropy(y_clean[right_indices]) if np.any(right_indices) else 0

        p_left = np.sum(left_indices) / len(y_clean)
        p_right = np.sum(right_indices) / len(y_clean)

        split_entropy = p_left*left_entropy + p_right*right_entropy
        gain = parent_entropy - split_entropy

        if gain > best_gain:
            best_gain = gain
            best_value = val

    return best_gain, best_value


def gain_ratio_nominal(X_col, y, parent_entropy):
    """
    Compute the gain ratio for a nominal/categorical feature (binary split).
    Gain Ratio = IG / split_info
    """
    not_missing_mask = ~pd.isna(X_col)
    X_col_clean = X_col[not_missing_mask]
    y_clean = y[not_missing_mask]

    if len(X_col_clean) == 0:
        return 0.0, None

    unique_values = np.unique(X_col_clean)
    if len(unique_values) <= 1:
        return 0.0, unique_values[0] if len(unique_values) == 1 else None

    best_ratio = -1.0
    best_value = None
    parent_enth = calculate_entropy(y_clean)

    for val in unique_values:
        left_indices = (X_col_clean == val)
        right_indices = (X_col_clean != val)

        left_entropy = calculate_entropy(y_clean[left_indices]) if np.any(left_indices) else 0
        right_entropy = calculate_entropy(y_clean[right_indices]) if np.any(right_indices) else 0

        p_left = np.sum(left_indices) / len(y_clean)
        p_right = np.sum(right_indices) / len(y_clean)

        ig = parent_enth - (p_left*left_entropy + p_right*right_entropy)

        # SplitInfo
        split_info = 0.0
        if p_left > 0:
            split_info -= p_left * np.log2(p_left)
        if p_right > 0:
            split_info -= p_right * np.log2(p_right)

        if split_info > 0:
            ratio = ig / split_info
        else:
            ratio = 0.0

        if ratio > best_ratio:
            best_ratio = ratio
            best_value = val

    return best_ratio, best_value

def gini_nominal(X_col, y, parent_gini):
    """
    Compute the Gini decrease for a nominal/categorical feature (binary split).
    Gini Decrease = parent_gini - weighted child gini
    """
    not_missing_mask = ~pd.isna(X_col)
    X_col_clean = X_col[not_missing_mask]
    y_clean = y[not_missing_mask]

    if len(X_col_clean) == 0:
        return 0.0, None

    unique_values = np.unique(X_col_clean)
    if len(unique_values) <= 1:
        return 0.0, unique_values[0] if len(unique_values) == 1 else None

    best_gain = -1.0
    best_value = None

    for val in unique_values:
        left_indices = (X_col_clean == val)
        right_indices = (X_col_clean != val)

        left_gini = calculate_gini(y_clean[left_indices]) if np.any(left_indices) else 0
        right_gini = calculate_gini(y_clean[right_indices]) if np.any(right_indices) else 0

        p_left = np.sum(left_indices) / len(y_clean)
        p_right = np.sum(right_indices) / len(y_clean)

        child_gini = p_left*left_gini + p_right*right_gini
        gain = parent_gini - child_gini

        if gain > best_gain:
            best_gain = gain
            best_value = val

    return best_gain, best_value


def check_feature_vector_is_categorical(arr):
    """
    Checks whether a NumPy array contains any string values, which would
    indicate that the feature vector is categorical (since categorical features
    are typically represented as strings in this case).

    Parameters:
    -----------
    arr : numpy.ndarray
        A NumPy array with dtype 'object', where elements can be either float or string.

    Returns:
    --------
    bool
        True if any string is found in the array (indicating categorical data),
        False otherwise.
    """
    contains_string = np.any(np.vectorize(lambda x: isinstance(x, str) and x != np.nan)(arr))
    return contains_string


class ID3(DecisionTree):
    """
    ID3 algorithm with:
      - Option to use Information Gain, Gain Ratio, or Gini Impurity
      - A flexible penalty for high-missing-rate features

    In classical ID3:
      - We do multi-way splits for nominal features.
      - Here, we do a binary split (True/False branches) to match
        the base DecisionTree's left_child/right_child structure.
      - For continuous features, we find the best threshold.
    """

    def __init__(
        self,
        stopping_criteria=None,
        max_depth=None,
        min_samples_split=2,
        min_impurity_decrease=0.0,
        criterion="information_gain",    # "information_gain" | "gain_ratio" | "gini"
        missing_penalty_power=1.0        # exponent for penalizing fraction of non-missing
    ):
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
            Minimum info gain (or gini decrease) needed to allow a split.
        criterion : str
            "information_gain", "gain_ratio", or "gini" to control the splitting measure.
        missing_penalty_power : float
            Exponent used to penalize the fraction of non-missing samples for a feature.
            If fraction_nonmissing = f, raw_gain = g, we do:
                penalized_gain = g * (f^missing_penalty_power).
            Larger exponents impose stronger penalties on high-missing-rate features.
        """
        super().__init__(
            decision_criteria=self._best_split,
            stopping_criteria=stopping_criteria,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_impurity_decrease=min_impurity_decrease
        )
        self.criterion = criterion
        self.missing_penalty_power = missing_penalty_power

    def _compute_gain(
        self,
        X_col: np.ndarray,
        y: np.ndarray,
        feature_index: int
    ):
        """
        Compute the split 'gain' (information gain, gain ratio, or gini decrease)
        for the given column, returning (best_gain, best_threshold_or_value).

        For continuous features, return threshold; for nominal, return the chosen category.
        """
        not_missing_mask = ~np.isnan(X_col) if np.issubdtype(X_col.dtype, np.number) \
                             else np.array([val is not None for val in X_col])
        fraction_nonmissing = np.sum(not_missing_mask) / len(X_col)

        if not check_feature_vector_is_categorical(X_col):
            # Continuous
            if self.criterion == "information_gain":
                parent_entropy = calculate_entropy(y)
                gain, threshold = information_gain_continuous(X_col, y, parent_entropy)
                return gain, threshold

            elif self.criterion == "gain_ratio":
                # parent_entropy needed inside the function
                ratio, threshold = gain_ratio_continuous(X_col, y, parent_entropy=None)
                # We do not pass parent_entropy since gain_ratio_continuous internally recalculates
                return ratio, threshold

            elif self.criterion == "gini":
                parent_g = calculate_gini(y)
                gain, threshold = gini_continuous(X_col, y, parent_g)
                return gain, threshold

            else:
                raise ValueError(f"Unknown criterion: {self.criterion}")

        else:
            # Nominal
            if self.criterion == "information_gain":
                parent_entropy = calculate_entropy(y)
                gain, chosen_val = information_gain_nominal(X_col, y, parent_entropy)
                return gain, chosen_val

            elif self.criterion == "gain_ratio":
                ratio, chosen_val = gain_ratio_nominal(X_col, y, parent_entropy=None)
                return ratio, chosen_val

            elif self.criterion == "gini":
                parent_g = calculate_gini(y)
                gain, chosen_val = gini_nominal(X_col, y, parent_g)
                return gain, chosen_val

            else:
                raise ValueError(f"Unknown criterion: {self.criterion}")

    def _best_split(self, X, y):
        """
        Finds the best split for the given dataset based on the chosen criterion
        ("information_gain", "gain_ratio", or "gini"), penalizing features with high missing rate.

        Returns
        -------
        dict
            {
                "impurity_decrease": float,  # Best penalized measure
                "criterion": callable,       # A function x -> True/False for the best split
                "feature_idx": int,          # Index of the best feature
                "threshold": float or object # For continuous: threshold
                                             # For nominal: the chosen category
            }
        """
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        best_criterion = None

        n_samples, n_features = X.shape

        for feature_index in range(n_features):
            feature_values = X[:, feature_index]
            # fraction of non-missing
            not_missing_mask = ~pd.isna(feature_values)
            fraction_nonmissing = np.sum(not_missing_mask) / n_samples
            if fraction_nonmissing == 0:
                continue
            else:
                # 1) Compute unpenalized gain (IG, ratio, or gini decrease)
                raw_gain, threshold_or_value = self._compute_gain(
                    feature_values, y, feature_index
                )

                # 2) Apply penalty: multiply by fraction_nonmissing^missing_penalty_power
                penalized_gain = raw_gain * (fraction_nonmissing ** self.missing_penalty_power)

                # 3) Track the best
                if penalized_gain > best_gain:
                    best_gain = penalized_gain
                    best_feature = feature_index
                    best_threshold = threshold_or_value

                    # For continuous
                    if not check_feature_vector_is_categorical(feature_values):
                        if threshold_or_value is not None:
                            best_criterion = lambda row, thr=threshold_or_value, f=feature_index: (
                                (not self._is_missing(row[f])) and (row[f] <= thr)
                            )
                        else:
                            best_criterion = None
                    else:
                        # Nominal
                        if threshold_or_value is not None:
                            best_criterion = lambda row, val=threshold_or_value, f=feature_index: (
                                (not self._is_missing(row[f])) and (row[f] == val)
                            )
                        else:
                            best_criterion = None

        # Return dictionary in the format expected by the base DecisionTree
        return {
            "impurity_decrease": best_gain if best_gain > 0 else 0.0,
            "criterion": best_criterion,
            "feature_idx": best_feature,
            "threshold": best_threshold
        }
