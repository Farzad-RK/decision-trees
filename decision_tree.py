import numpy as np
from graphviz import Digraph
from sklearn.base import BaseEstimator, ClassifierMixin
from node import Node


class DecisionTree(BaseEstimator, ClassifierMixin):
    def __init__(
            self,
            decision_criteria=None,
            stopping_criteria=None,
            max_depth=None,
            min_samples_split=2,
            min_impurity_decrease=0.0,
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
        self.root = self._build_tree(X, y, depth=0, parent_label=None)

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

    def _build_tree(self, X, y, depth, parent_label=None):
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

        Note: we also accept 'parent_label' to handle empty y by using the parent's majority label.
        """
        # If there are no samples in y, directly create a leaf with the parent's label
        if len(y) == 0:
            leaf = Node(is_leaf=True)
            # Fallback: if parent_label is None, default to 0 or any fixed class
            fallback_label = parent_label if parent_label is not None else 0
            leaf.set_leaf(label=fallback_label)
            return leaf

        unique_labels = np.unique(y)

        # Leaf condition checks:
        if (
                len(unique_labels) == 1
                or len(y) < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)
        ):
            leaf = Node(is_leaf=True)
            # Normal leaf: pick majority from this node's subset
            majority_label = np.bincount(y).argmax()
            leaf.set_leaf(label=majority_label)
            return leaf

        # Attempt to find best split
        if not self.decision_criteria:
            raise ValueError("No decision_criteria function provided.")

        best_split = self.decision_criteria(X, y)
        if best_split['impurity_decrease'] < self.min_impurity_decrease:
            # Convert to leaf
            leaf = Node(is_leaf=True)
            majority_label = np.bincount(y).argmax()
            leaf.set_leaf(label=majority_label)
            return leaf

        # Create internal node
        node = Node(
            is_leaf=False,
            decision_criterion=best_split['criterion'],
            feature_idx=best_split.get('feature_idx', None),
            threshold=best_split.get('threshold', None)
        )

        # Split data into left / right / missing
        feature_idx = node.feature_idx
        left_indices = []
        right_indices = []
        missing_indices = []

        for i, row in enumerate(X):
            if self._is_missing(row[feature_idx]):
                missing_indices.append(i)
            else:
                if best_split['criterion'](row):
                    left_indices.append(i)
                else:
                    right_indices.append(i)

        left_indices = np.array(left_indices, dtype=int)
        right_indices = np.array(right_indices, dtype=int)
        missing_indices = np.array(missing_indices, dtype=int)

        num_left = len(left_indices)
        num_right = len(right_indices)
        total_nonmissing = num_left + num_right

        if total_nonmissing == 0:
            node.missing_left_fraction = 1.0
            left_indices = np.concatenate([left_indices, missing_indices])
            missing_indices = []
        else:
            left_fraction = num_left / total_nonmissing
            node.missing_left_fraction = left_fraction
            if left_fraction >= 0.5:
                left_indices = np.concatenate([left_indices, missing_indices])
            else:
                right_indices = np.concatenate([right_indices, missing_indices])

        X_left, y_left = X[left_indices], y[left_indices]
        X_right, y_right = X[right_indices], y[right_indices]

        # Recursively build subtrees
        # The 'majority_label' for these children is the current node's majority if it were to become a leaf
        # so we pass that as 'parent_label' in case the child's y is empty.
        node_majority = np.bincount(y).argmax()
        left_child = self._build_tree(X_left, y_left, depth + 1, parent_label=node_majority)
        right_child = self._build_tree(X_right, y_right, depth + 1, parent_label=node_majority)
        node.set_children(left_child, right_child)

        # Track feature importance
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

    def cost_complexity_prune(self, X, y, alpha=0.0):
        """
        Perform a single or iterative cost-complexity pruning with parameter alpha.

        - If alpha=0, we effectively remove any splits that don't reduce training error.
        - If alpha>0, we trade off error vs. tree size.

        This is a simplified version of CART's approach.
        Typically, we'd do a sequence of prunes and pick the best by validation,
        but here we just do repeated pruning while beneficial.

        Parameters
        ----------
        X : np.ndarray
            Training features or a separate pruning set.
        y : np.ndarray
            Labels (same length as X).
        alpha : float
            The complexity penalty. Higher alpha means heavier penalty on # leaves.

        Returns
        -------
        None (the tree is pruned in-place).
        """

        # We'll define some helpers inside.

        def _postorder_nodes(node):
            """Return a list of internal nodes in post-order (children before parent)."""
            nodes = []
            if node is None or node.is_leaf:
                return nodes
            # Recur left
            if node.left_child is not None:
                nodes.extend(_postorder_nodes(node.left_child))
            # Recur right
            if node.right_child is not None:
                nodes.extend(_postorder_nodes(node.right_child))
            # Then add this node
            nodes.append(node)
            return nodes

        def _count_leaves(node):
            """Count how many leaf nodes are in this subtree."""
            if node.is_leaf:
                return 1
            return _count_leaves(node.left_child) + _count_leaves(node.right_child)

        def _compute_subtree_error(node, X_sub, y_sub):
            """Number of misclassifications for all samples (X_sub, y_sub) in this subtree."""
            if len(y_sub) == 0:
                # no samples => no immediate error
                return 0
            preds = []
            for i in range(len(y_sub)):
                preds.append(self._predict_sample(X_sub[i], node))
            preds = np.array(preds)
            return np.sum(preds != y_sub)

        def _gather_samples_for_subtree(node, X_sub, y_sub):
            """
            Return the subset of (X_sub, y_sub) that fall under the subtree rooted at 'node'.
            We do a BFS or DFS from 'node' -> but it's easier to just check which samples predict
            to a leaf in that subtree. Then we keep them.
            """
            # One approach:
            # We can reuse predict logic, but we want to see which end node is under "node".
            # We'll do an index-based approach: if the path from the root includes 'node' as an ancestor,
            # we skip or keep. This can be complicated.
            # Simpler approach is to see which samples would be predicted
            # if we replaced the entire root with 'node'.
            # But that breaks the overall tree structure.
            #
            # Alternatively, if we have a function that classifies from 'node' downward (like _predict_sample but starting here),
            # then any sample that calls node or its children is "in" this subtree.
            # But from a whole-tree perspective, some samples might not reach 'node' unless we do partial checks.
            #
            # EASIEST: We want the training examples that definitely belong to the subtree "node" in the normal tree structure.
            # We can trace from the root down.
            # => We'll define a helper that from the root, for each sample, we track the path of nodes.
            # If it includes 'node', we add that sample to the set.
            #
            # For large data, this might be slow, but it's simpler logically.

            included_indices = []
            for i, row in enumerate(X_sub):
                # follow normal predict path, collecting visited nodes
                current = self.root
                visited = []
                while not current.is_leaf:
                    visited.append(current)
                    if self._is_missing(row[current.feature_idx]):
                        # go left or right based on missing_left_fraction
                        if current.missing_left_fraction is None or current.missing_left_fraction >= 0.5:
                            current = current.left_child
                        else:
                            current = current.right_child
                    else:
                        decision = current.evaluate(row)
                        if decision:
                            current = current.left_child
                        else:
                            current = current.right_child
                    if current is None:
                        break
                # Also add the final leaf
                if current is not None:
                    visited.append(current)

                if node in visited:
                    included_indices.append(i)

            included_indices = np.array(included_indices, dtype=int)
            return X_sub[included_indices], y_sub[included_indices]

        def _prune_node(node, majority_label):
            """
            Turn 'node' into a leaf with the given majority_label.
            """
            node.is_leaf = True
            node.label = majority_label
            node.left_child = None
            node.right_child = None
            node.decision_criterion = None
            node.feature_idx = None
            node.threshold = None
            node.missing_left_fraction = None

        # Now we do repeated "weakest link" pruning until no improvement or tree collapses
        while True:
            nodes = _postorder_nodes(self.root)
            if not nodes:
                break  # no internal nodes left to prune

            best_node = None
            best_alpha_gain = float('inf')  # the smallest alpha = deltaR / deltaLeaves
            best_deltaR = 0
            best_deltaLeaves = 1
            best_majority_label = 0

            # We'll track the entire R(T) for the root first
            full_error = _compute_subtree_error(self.root, X, y)
            full_leaf_count = _count_leaves(self.root)

            for node in nodes:
                # Subtree T_node
                X_subtree, y_subtree = _gather_samples_for_subtree(node, X, y)
                # current subtree's error
                unpruned_error = _compute_subtree_error(node, X_subtree, y_subtree)
                subtree_leaves = _count_leaves(node)

                # If we prune this node into a leaf, what's the new error for that subtree?
                # The leaf's label would typically be the majority of y_subtree (if it's not empty)
                if len(y_subtree) > 0:
                    pruned_label = np.bincount(y_subtree).argmax()
                else:
                    # fallback to node.label or 0
                    pruned_label = node.label if node.label is not None else 0

                # If we prune, the subtree has 1 leaf => pruned_error
                pruned_error = np.sum(y_subtree != pruned_label)

                # delta R = (unpruned_error -> pruned_error)
                # But we want the difference in error for the *whole tree*, not just the subtree,
                # so let's define that carefully:
                #   old tree error portion = unpruned_error
                #   new tree error portion = pruned_error
                # deltaR_subtree = pruned_error - unpruned_error
                #
                # For cost complexity, we often do node-based increments, but let's do the local difference:
                delta_error = (pruned_error - unpruned_error)

                # Leaves difference: subtree_leaves -> 1
                delta_leaves = subtree_leaves - 1

                # local alpha
                # alpha_node = (R_unpruned - R_pruned) / (L_pruned_removed)
                # but we have to be consistent with sign:
                # Actually we want the cost difference if we prune:
                # cost(T_pruned) - cost(T_unpruned) = delta_error + alpha*(1 - subtree_leaves)
                # => alpha_node = delta_error / delta_leaves  (since delta_leaves < 0 if subtree_leaves>1)
                # but typically we consider positive alpha => we take absolute ratio
                # We'll handle the typical formula: alpha_node = (unpruned_error - pruned_error) / (subtree_leaves - 1),
                # which is how CART finds "weakest link".
                #
                # We want alpha_node >= 0. If pruned_error > unpruned_error => alpha_node < 0 => not beneficial
                # so skip if it doesn't reduce error. We'll still compute to see if it's minimal positive though.

                # The "gain" from pruning is (unpruned_error - pruned_error).
                gain_in_error = unpruned_error - pruned_error  # how much error we reduce
                if delta_leaves == 0:  # edge case: subtree leaves = 1 => can't prune further
                    continue

                alpha_node = gain_in_error / delta_leaves  # might be negative if it hurts accuracy

                # We'll pick the "weakest link" => smallest alpha_node
                # but only if alpha_node >= 0. If alpha_node < 0 => pruning actually increases error, skip it.
                if alpha_node < 0:
                    continue

                # We compare alpha_node to alpha because we want to prune nodes that have alpha_node <= alpha
                # i.e. it's cheap to prune them. Among those, pick the smallest alpha_node as the "weakest link".
                if alpha_node <= alpha and alpha_node < best_alpha_gain:
                    print(alpha_node)
                    best_alpha_gain = alpha_node
                    best_node = node
                    best_deltaR = gain_in_error
                    best_deltaLeaves = delta_leaves
                    best_majority_label = pruned_label

            # If we didn't find any node to prune for alpha, we break
            if best_node is None:
                break

            # Otherwise, prune that node
            _prune_node(best_node, best_majority_label)

        # End while
        # Now the tree is pruned in place

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
