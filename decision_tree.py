import numpy as np
from graphviz import Digraph
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from node import Node
import pandas as pd


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            decision_criteria=None,
            stopping_criteria=None,
            max_depth=None,
            min_samples_split=2,
            min_impurity_decrease=0.0,
            continuous_features_indexes=None
    ):
        """
        A flexible DecisionTree classifier that follows scikit-learn's API.

        Parameters
        ----------
        decision_criteria : callable or None
            A function (externally defined) for selecting a split.
            Expected to return a dictionary with keys:
              - 'criterion': callable -> the actual splitting function
                              e.g. lambda x: x[feature_idx] < threshold
              - 'impurity_decrease': float -> the impurity decrease for that split
              - 'feature_idx': int -> the feature index used in the split
              - 'threshold': float -> the numeric threshold used

        stopping_criteria : dict or None
            Additional stopping criteria.
            E.g. {'max_depth': 5, 'min_samples_split': 2, 'min_impurity_decrease': 0.0}

        max_depth : int or None
            Maximum tree depth. If None, no limit.

        min_samples_split : int
            Minimum samples needed to split an internal node.

        min_impurity_decrease : float
            Minimum impurity decrease needed to justify a split.
        """
        self.decision_criteria = decision_criteria
        self.stopping_criteria = stopping_criteria
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease

        self.root = None  # Will store the root Node after fitting

        """
        To keep track of the importance of each feature
        This attribute is going to be calculated after the fit has been called
        """
        self.feature_importances_ = None
        self.continuous_features_indexes = continuous_features_indexes

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training feature matrix
        y : np.ndarray of shape (n_samples,)
            Training labels

        Returns
        -------
        self : object
            Fitted estimator
        """
        # If stopping_criteria was provided, override any parameters
        if self.stopping_criteria is not None:
            self.max_depth = self.stopping_criteria.get('max_depth', self.max_depth)
            self.min_samples_split = self.stopping_criteria.get(
                'min_samples_split',
                self.min_samples_split
            )
            self.min_impurity_decrease = self.stopping_criteria.get(
                'min_impurity_decrease',
                self.min_impurity_decrease
            )
        # Initializing/Resetting the feature importance tracker
        self._feature_importance_tracker = np.zeros(X.shape[1], dtype=float)

        # Build the tree recursively
        self.root = self._build_tree(X, y, depth=0)

        # after building, normalize importance
        total = self._feature_importance_tracker.sum()
        if total > 0:
            self.feature_importances_ = self._feature_importance_tracker / total
        else:
            # e.g., all samples same class => no splits => no importance
            self.feature_importances_ = np.zeros(X.shape[1], dtype=float)

        return self

    def _is_missing(self, val):
        """
        Check if a value is missing.
        For numeric features, we treat np.nan as missing.
        For nominal features, we treat None as missing
        """
        if isinstance(val, float) and np.isnan(val):
            return True
        if val is None:
            return True
        return False

    def _build_tree(self, X, y, depth):
        """
        Recursively builds the decision tree.

        Missing-Value Handling (extension):
        -----------------------------------
        1. For finding the best feature (decision_criteria), we only use
           non-missing samples for that candidate feature.
        2. After the best feature is chosen, we do a "hard assignment" of
           missing-feature samples to the left or right branch:
              - We first split the non-missing samples in the usual way.
              - We calculate the fraction of non-missing samples going left
                vs. right.
              - node.missing_left_fraction = (# going left / # non-missing)
              - All missing samples go to the branch that has the majority
                of non-missing samples (or to left if equal).
        3. At prediction time, if the feature value is missing for that node,
           we check node.missing_left_fraction to see where to send the sample.
        """
        # Check if this node should be a leaf:
        #  1) All labels are identical
        #  2) Not enough samples to split
        #  3) Max depth reached
        unique_labels = np.unique(y)
        if (
                len(unique_labels) == 1
                or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
        ):
            leaf = Node(is_leaf=True)
            leaf.set_leaf(label=np.bincount(y).argmax())  # majority class
            return leaf

        # Attempt to find the best split using the user-defined criterion
        if not self.decision_criteria:
            # Fallback: If no external decision_criteria is provided,
            raise ValueError(
                "No decision_criteria function provided. "
                "Please provide a callable for splitting logic."
            )

        best_split = self.decision_criteria(X, y)
        if best_split['impurity_decrease'] < self.min_impurity_decrease:
            # Not enough impurity decrease, convert to leaf
            leaf = Node(is_leaf=True)
            leaf.set_leaf(label=np.bincount(y).argmax())
            return leaf

        # Create internal node
        node = Node(
            is_leaf=False,
            decision_criterion=best_split['criterion'],  # the lambda that does x[feature_idx] < threshold
            feature_idx=best_split.get('feature_idx', None),
            threshold=best_split.get('threshold', None)
        )

        # Split the data into left and right branches for non-missing values
        feature_idx = node.feature_idx
        left_indices = []
        right_indices = []
        missing_indices = []

        for i, row in enumerate(X):
            # Check if the chosen feature is missing
            if self._is_missing(row[feature_idx]):
                missing_indices.append(i)
            else:
                # Normal splitting
                if best_split['criterion'](row):
                    left_indices.append(i)
                else:
                    right_indices.append(i)

        left_indices = np.array(left_indices, dtype=int)
        right_indices = np.array(right_indices, dtype=int)
        missing_indices = np.array(missing_indices, dtype=int)

        # Distribute missing data in a "hard assignment" fashion
        # based on the proportion of non-missing samples that go left/right
        num_left = len(left_indices)
        num_right = len(right_indices)
        total_nonmissing = num_left + num_right

        if total_nonmissing == 0:
            # Edge case: if *all* samples are missing for this feature,
            # just send them all left by default and set fraction = 1.0
            node.missing_left_fraction = 1.0
            left_indices = np.concatenate([left_indices, missing_indices])
            missing_indices = []
        else:
            left_fraction = num_left / total_nonmissing
            node.missing_left_fraction = left_fraction
            # Decide where missing samples go:
            if left_fraction >= 0.5:
                left_indices = np.concatenate([left_indices, missing_indices])
            else:
                right_indices = np.concatenate([right_indices, missing_indices])

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Recursively build left and right subtrees
        node.set_children(
            self._build_tree(X_left, y_left, depth + 1),
            self._build_tree(X_right, y_right, depth + 1)
        )

        # Updating feature tracker to accumulate the feature score
        node_gain = best_split['impurity_decrease']
        node_feature = best_split['feature_idx']
        self._feature_importance_tracker[node_feature] += node_gain

        return node

    def predict(self, X):
        """
        Predict class labels for the samples in X.
        """
        return np.array([self._predict_sample(sample, self.root) for sample in X])

    def _predict_sample(self, sample, node):
        """
        Traverse the tree for a single sample.

        Extended to handle missing values:
          - If the node is non-leaf and the sample's feature is missing,
            we check node.missing_left_fraction:
               * If it's >= 0.5, predict down the left child
               * Else, go right child
        """
        if node.is_leaf:
            return node.label

        # If the feature is missing, direct the sample based on missing_left_fraction
        if self._is_missing(sample[node.feature_idx]):
            if node.missing_left_fraction is None:
                # If for any reason it's not set, default to left or handle gracefully
                return self._predict_sample(sample, node.left_child)
            else:
                if node.missing_left_fraction >= 0.5:
                    return self._predict_sample(sample, node.left_child)
                else:
                    return self._predict_sample(sample, node.right_child)
        else:
            # Evaluate at the current node and go left or right
            decision = node.evaluate(sample)
            if decision:  # True means go left
                return self._predict_sample(sample, node.left_child)
            else:
                return self._predict_sample(sample, node.right_child)

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.

        In scikit-learn, `score` for a classifier defaults to accuracy.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def visualize_tree(self, feature_names=None, file_name='decision_tree'):
        """
        Visualizes the decision tree using graphviz.

        Parameters
        ----------
        feature_names : list of str, optional
            A list of feature names to display at each internal node.
            If None, the feature index is used.
        file_name : str, optional
            The name of the rendered graphviz file
        Returns
        -------
        dot : graphviz.Digraph
            The graphviz Digraph object representing the tree.
        """
        dot = Digraph()
        self._add_node(dot, self.root, node_id="0", feature_names=feature_names)
        dot.render(f"./{file_name}", format="png")

    def _add_node(self, dot, node, node_id, feature_names=None):
        """
        Recursively add nodes and edges to the graphviz Digraph.
        """
        if node.is_leaf:
            # Leaf node -> label with class
            dot.node(node_id, label=f"Leaf\nClass={node.label}", shape="box")
        else:
            # Internal node
            # Show "feature < threshold" if known
            if node.feature_idx is not None and node.threshold is not None:
                feat_label = (feature_names[node.feature_idx]
                              if feature_names
                              else f"X[{node.feature_idx}]")
                # Check if threshold is numeric before applying format code 'f'
                if isinstance(node.threshold, (int, float)):
                    node_label = f"{feat_label} < {node.threshold:.5f}"
                else:
                    # If threshold is not numeric (likely categorical), format differently
                    node_label = f"{feat_label} == {node.threshold}"
            else:
                # fallback if not provided
                node_label = "Split Node"
            dot.node(node_id, label=node_label, shape="ellipse")

            # Add left child
            left_id = node_id + "0"
            dot.edge(node_id, left_id, label="True")
            self._add_node(dot, node.left_child, left_id, feature_names)

            # Add right child
            right_id = node_id + "1"
            dot.edge(node_id, right_id, label="False")
            self._add_node(dot, node.right_child, right_id, feature_names)

    def get_params(self, deep=True):
        """
        Return estimator parameters for this DecisionTree (used by scikit-learn).
        """
        return {
            "continuous_features_indexes": self.continuous_features_indexes,
            "stopping_criteria": self.stopping_criteria,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_impurity_decrease": self.min_impurity_decrease,
        }

    def set_params(self, **params):
        """
        Set the parameters of this estimator.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self):
        return f"DecisionTree(max_depth={self.max_depth}, min_samples_split={self.min_samples_split})"
