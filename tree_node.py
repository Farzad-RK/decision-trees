

class TreeNode:
    def __init__(self, decision_criterion=None, is_leaf=False, left_child=None, right_child=None):
        """
        Initializes a tree node.

        Parameters:
        - decision_criterion (callable): A function that takes a numpy vector (data point) as input
          and returns a Boolean value as output. This represents the splitting criterion.
        - is_leaf (bool): Flag to indicate if the node is a leaf.
        - left_child (TreeNode): The left child node.
        - right_child (TreeNode): The right child node.
        """
        self.decision_criterion = decision_criterion  # Function to test decision criterion
        self.is_leaf = is_leaf                        # Whether the node is a leaf
        self.left_child = left_child                        # Left child (TreeNode)
        self.right_child = right_child                       # Right child (TreeNode)
        self.label = None                             # Class label for leaf nodes

    def set_children(self, left_child, right_child):
        """
        Sets the left and right children of the node.

        Parameters:
        - left_child (TreeNode): The left child node.
        - right_child (TreeNode): The right child node.
        """
        self.left_child = left_child
        self.right_child = right_child

    def set_leaf(self, label):
        """
        Converts the node to a leaf and assigns a label.

        Parameters:
        - label: The class label to assign to the leaf.
        """
        self.is_leaf = True
        self.label = label

    def evaluate(self, sample):
        """
        Evaluates the decision criterion for a given sample.

        Parameters:
        - sample (numpy.ndarray): A single feature vector (data point).

        Returns:
        - bool: The result of the decision criterion (True for left, False for right).
        """
        if self.decision_criterion is None:
            raise ValueError("Decision criterion is not set for this node.")
        return self.decision_criterion(sample)

    def __repr__(self):
        """
        String representation of the node.

        Returns:
        - str: Description of the node.
        """
        if self.is_leaf:
            return f"Leaf Node(label={self.label})"
        else:
            return f"Internal Node(decision_criterion={self.decision_criterion})"