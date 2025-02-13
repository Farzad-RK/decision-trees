import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class RandomForest(BaseEstimator, ClassifierMixin):
    """
    A basic RandomForest classifier that uses the provided DecisionTree
    (e.g., your ID3 class) as the base_estimator.
    """

    def __init__(
            self,
            base_estimator_class,
            n_estimators=10,
            max_features="sqrt",  # can be int, float, "sqrt", or "log2", etc.
            random_state=None,
            **base_estimator_params
    ):
        """
        Parameters
        ----------
        base_estimator_class : class
            The class of the Decision Tree (e.g. ID3) you want to use.
        n_estimators : int
            Number of trees in the forest.
        max_features : int, float or str
            Number of features to consider when looking for the best split.
            - If int, consider `max_features` features.
            - If float, consider `max_features * n_features` features.
            - If "sqrt", consider `sqrt(n_features)`.
            - If "log2", consider `log2(n_features)`.
            - If None, consider all features.
        random_state : int or None
            Random seed for reproducibility.
        **base_estimator_params : any
            Additional parameters passed to the base_estimator constructor.
        """
        self.base_estimator_class = base_estimator_class
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.random_state = random_state
        self.base_estimator_params = base_estimator_params

        self.estimators_ = []  # list of fitted trees
        self.features_subsets_ = []  # each tree's chosen subset of features
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _get_feature_subset(self, n_features):
        """
        Determine which features to sample for each tree, given max_features.
        """
        if isinstance(self.max_features, int):
            k = self.max_features
        elif isinstance(self.max_features, float):
            k = int(self.max_features * n_features)
        elif self.max_features == "sqrt":
            k = int(np.sqrt(n_features))
        elif self.max_features == "log2":
            k = int(np.log2(n_features))
        elif self.max_features is None:
            k = n_features
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

        # Sample k features out of n_features
        return np.random.choice(n_features, size=k, replace=False)

    def fit(self, X, y):
        """
        Build a random forest of base_estimator_class trees from the training set (X, y).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,)

        Returns
        -------
        self
        """
        n_samples, n_features = X.shape

        self.estimators_ = []
        self.features_subsets_ = []

        for _ in range(self.n_estimators):
            # Bootstrap sample the training data
            bootstrap_indices = np.random.randint(0, n_samples, size=n_samples)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Random subset of features
            features_subset = self._get_feature_subset(n_features)
            self.features_subsets_.append(features_subset)

            # Create a new base estimator
            tree = self.base_estimator_class(**self.base_estimator_params)

            # Fit using only the selected features
            tree.fit(X_bootstrap[:, features_subset], y_bootstrap)
            self.estimators_.append(tree)

        return self

    def predict(self, X):
        """
        Predict class for X by majority voting among all trees.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
        """
        # Collect predictions from each tree
        all_preds = []
        for tree, feats in zip(self.estimators_, self.features_subsets_):
            preds = tree.predict(X[:, feats])
            all_preds.append(preds)

        # all_preds is a list of arrays, each shape (n_samples,)
        # Stack horizontally => shape (n_estimators, n_samples)
        all_preds = np.vstack(all_preds)

        # Majority vote for each sample
        from scipy.stats import mode
        # mode returns (mode_values, count), axis=0 for columns => we want index [0] for the actual mode
        y_pred, _ = mode(all_preds, axis=0, keepdims=True)
        y_pred = y_pred.flatten().astype(int)

        return y_pred

    def score(self, X, y):
        """
        Return the mean accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
