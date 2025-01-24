import numpy as np
from node import Node
from sklearn.model_selection import StratifiedKFold
from graphviz import Digraph


class DecisionTree:
    def __init__(self, decision_criteria=None, stopping_criteria=None):
        """
        Initializes the tree predictor.

        Parameters:
        - decision_criteria (callable): Function or strategy to select the best decision criterion for a node.
        - stopping_criteria (dict): Dictionary defining conditions to halt tree growth. Keys could include:
            - 'max_depth': Maximum depth of the tree.
            - 'min_samples_split': Minimum number of samples required to split a node.
            - 'min_impurity_decrease': Minimum decrease in impurity required for a split.
        """
        self.root = None
        self.decision_criteria = decision_criteria
        self.stopping_criteria = stopping_criteria
        self.max_depth = stopping_criteria.get('max_depth', None) if stopping_criteria else None
        self.min_samples_split = stopping_criteria.get('min_samples_split', 2) if stopping_criteria else 2
        self.min_impurity_decrease = stopping_criteria.get('min_impurity_decrease', 0.0) if stopping_criteria else 0.0

    def train(self, X, y):
        """
        Trains the decision tree on the given dataset.

        Parameters:
        - X (numpy.ndarray): Training feature matrix of shape (n_samples, n_features).
        - y (numpy.ndarray): Training labels of shape (n_samples,).
        """
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Recursively builds the decision tree.

        Parameters:
        - X (numpy.ndarray): Feature matrix at the current node.
        - y (numpy.ndarray): Labels at the current node.
        - depth (int): Current depth of the tree.

        Returns:
        - TreeNode: Root node of the subtree.
        """
        # Check stopping criteria
        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth and depth >= self.max_depth):
            leaf = Node(is_leaf=True)
            leaf.set_leaf(label=np.bincount(y).argmax())  # Set majority class label
            return leaf

        # Select the best split (decision criterion)
        best_split = self.decision_criteria(X, y)  # This should be defined externally for specific algorithms
        if best_split['impurity_decrease'] < self.min_impurity_decrease:
            leaf = Node(is_leaf=True)
            leaf.set_leaf(label=np.bincount(y).argmax())
            return leaf

        # Create an internal node
        node = Node(decision_criterion=best_split['criterion'])

        # Split the data
        left_indices = [i for i in range(len(X)) if best_split['criterion'](X[i])]
        right_indices = [i for i in range(len(X)) if not best_split['criterion'](X[i])]

        # Recursively build left and right children
        node.set_children(
            self._build_tree(X[left_indices], y[left_indices], depth + 1),
            self._build_tree(X[right_indices], y[right_indices], depth + 1)
        )

        return node

    def predict(self, X):
        """
        Predicts the labels for the given feature matrix.

        Parameters:
        - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).

        Returns:
        - numpy.ndarray: Predicted labels of shape (n_samples,).
        """
        return np.array([self._predict_sample(x, self.root) for x in X])

    def _predict_sample(self, sample, node):
        """
        Predicts the label for a single sample by traversing the tree.

        Parameters:
        - sample (numpy.ndarray): Single feature vector.
        - node (TreeNode): Current node in the tree.

        Returns:
        - int: Predicted label.
        """
        if node.is_leaf:
            return node.label
        if node.evaluate(sample):
            return self._predict_sample(sample, node.left_child)
        else:
            return self._predict_sample(sample, node.right_child)

    def evaluate(self, X, y):
        """
        Evaluates the decision tree on the given test dataset.

        Parameters:
        - X (numpy.ndarray): Test feature matrix of shape (n_samples, n_features).
        - y (numpy.ndarray): True labels of shape (n_samples,).

        Returns:
        - float: 0-1 loss (error rate).
        """
        predictions = self.predict(X)
        return np.mean(predictions != y)

    def cross_validate(self, X, y, k):
        """
        Performs stratified K-fold cross-validation.

        Parameters:
        - X (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        - y (numpy.ndarray): Labels of shape (n_samples,).
        - k (int): Number of folds.

        Returns:
        - float: Average 0-1 loss across all folds for training
        - float: Average 0-1 loss across all folds for validation
        """
        train_losses = []
        test_losses = []

        skf = StratifiedKFold(n_splits=k, shuffle=True)
        for train_indices, test_indices in skf.split(X, y):
            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]
            self.train(X_train, y_train)
            # Loss calculation of the train set
            train_losses.append(self.evaluate(X_train, y_train))
            # Loss calculation of the test set
            test_losses.append(self.evaluate(X_test, y_test))

        return np.mean(train_losses), np.mean(test_losses)

    def __repr__(self):
        return "DecisionTree()"
