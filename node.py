

class Node:
    def __init__(self,
                 is_leaf=False,
                 label=None,
                 decision_criterion=None,
                 feature_idx=None,
                 threshold=None):
        """
        Represents a node in the decision tree.

        Parameters
        ----------
        is_leaf : bool
            Whether this node is a leaf.
        label : int, optional
            The class label if this is a leaf node.
        decision_criterion : callable, optional
            A function that takes a sample (1D array) and returns True/False
            for left or right split.
        feature_idx : int, optional
            The feature index used for threshold comparison (if known).
        threshold : float, optional
            The threshold value used in the decision function (if known).
        """
        self.is_leaf = is_leaf
        self.label = label
        self.decision_criterion = decision_criterion
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left_child = None
        self.right_child = None

    def set_leaf(self, label):
        self.is_leaf = True
        self.label = label

    def set_children(self, left_child, right_child):
        self.left_child = left_child
        self.right_child = right_child

    def evaluate(self, sample):
        """
        Evaluate the sample at this node's decision criterion.
        (Only valid for non-leaf nodes.)
        """
        if self.is_leaf:
            return None
        return self.decision_criterion(sample)

