# Decision Tree and Random Forest Implementation

## Overview
This project provides an implementation of decision tree-based classifiers, including:
- **ID3 Decision Tree**: A decision tree classifier based on the ID3 algorithm, supporting information gain, gain ratio, and Gini impurity as splitting criteria.
- **Random Forest**: A random forest classifier built using the decision tree implementation as the base estimator.

The models are **scikit-learn compatible**, meaning they follow the fit/predict API and can be integrated with existing machine learning workflows.

## Project Structure
- `decision_tree.py`: Base class for decision trees, defining core tree-building logic.
- `id3_tree.py`: Implements the ID3 algorithm as an extension of `DecisionTree`.
- `random_forest.py`: Implements a Random Forest using decision trees as weak learners.
- `node.py`: Defines the structure of individual nodes in the decision tree.
- `utils.py`: Contains helper functions such as categorical data imputation.
- `main.py`: Provides a terminal-based menu to interact with the models.
- `requirements.txt`: Lists dependencies needed to run the project.

## Dependencies
This project requires the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `graphviz`
- `ucimlrepo`
- `joblib`
- `tqdm`
- `scikit-optimize`

### Installing Dependencies
To install the required packages, run:
```bash
pip install -r requirements.txt
```

## Graphviz Installation
The project uses **Graphviz** for visualizing decision trees. If you encounter errors related to Graphviz, install it manually:

**Ubuntu/Debian:**
```bash
sudo apt install graphviz
```

**MacOS (Homebrew):**
```bash
brew install graphviz
```

**Windows:**
1. Download the installer from: [Graphviz website](https://graphviz.gitlab.io/)
2. Install Graphviz and add it to your system's PATH.

## Running the Project
Execute the following command to start the interactive menu:
```bash
python main.py
```

## Terminal-Based Menu
The main script (`main.py`) provides an interactive terminal-based menu with the following options:

1. **Visualize Dataset**: Displays dataset statistics and missing value distribution.
2. **Evaluate Model and Plot Results**: Trains an ID3 classifier and plots performance.
3. **Visualize Decision Trees**: Trains and visualizes decision trees for different splitting criteria.
4. **Visualize Pruned Decision Trees**: Performs cost-complexity pruning and visualizes the results.
5. **Perform Hyperparameter Tuning**: Optimizes ID3 hyperparameters using Bayesian search.
6. **Perform Random Forest Classification**: Trains and evaluates a Random Forest classifier.
7. **Exit**: Exits the program.

## Class Descriptions
### **Node (`node.py`)**
Defines a node in the decision tree, storing attributes such as:
- Whether it is a leaf
- The feature used for splitting
- The threshold for continuous splits
- Left and right child nodes

### **DecisionTree (`decision_tree.py`)**
A base class for decision tree implementations, supporting:
- Custom stopping criteria (max depth, min samples per split, impurity decrease)
- Handling of missing values
- Cost-complexity pruning
- Visualization using Graphviz

### **ID3 (`id3_tree.py`)**
An implementation of the ID3 algorithm that extends `DecisionTree`.
- Supports `information_gain`, `gain_ratio`, and `gini` as splitting criteria.
- Penalizes features with high missing values.
- Compatible with scikit-learn's `fit`, `predict`, and `score` methods.

#### ID3 Parameters:
- `max_depth`: Maximum depth of the tree (`None` for unlimited, or an integer value, e.g., `5`).
- `min_samples_split`: Minimum samples required to split a node (`default=2`).
- `min_impurity_decrease`: Minimum decrease in impurity to justify a split (`default=0.0`).
- `criterion`: Splitting criterion; possible values are `"information_gain"`, `"gain_ratio"`, and `"gini"` (`default="information_gain"`).
- `missing_penalty_power`: A penalty applied to features with missing values, controlling their importance in splits (`default=1.0`).

### **RandomForest (`random_forest.py`)**
A Random Forest classifier built using `DecisionTree` as the base estimator.
- Trains multiple trees on bootstrapped datasets.
- Uses feature subsetting to decorrelate trees.
- Implements majority voting for classification.

#### RandomForest Parameters:
- `base_estimator_class`: Class of decision tree used (`default=ID3`).
- `n_estimators`: Number of trees in the forest (`default=10`).
- `max_features`: Number of features to consider per split (`"sqrt"`, `"log2"`, an integer, or `None` to use all features, `default="sqrt"`).
- `max_depth`: Maximum depth of each tree (`None` for unlimited, or an integer value, e.g., `5`).
- `min_samples_split`: Minimum samples required to split a node (`default=2`).
- `missing_penalty_power`: A penalty for missing values (`default=1.5`).

## Example Usage
To train and evaluate an ID3 decision tree:
```python
from id3_tree import ID3
from sklearn.model_selection import train_test_split

# Load dataset
X, y = load_data()  # Replace with actual dataset loading
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train ID3
model = ID3(max_depth=5, criterion='information_gain')
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy}")
```

To train and evaluate a Random Forest:
```python
from random_forest import RandomForest

rf = RandomForest(base_estimator_class=ID3, n_estimators=10, max_depth=5)
rf.fit(X_train, y_train)
print("Random Forest Accuracy:", rf.score(X_test, y_test))
```

