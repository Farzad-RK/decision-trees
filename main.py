import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np
from utils import impute_categorical_mode
from id3_tree import ID3
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split, GridSearchCV
from random_forest import RandomForest
from joblib import Parallel, delayed
from tqdm import tqdm
from skopt import BayesSearchCV
from skopt.space import Categorical

random_state = 42


def visualize_dataset():
    # fetch dataset
    secondary_mushroom = fetch_ucirepo(id=848)

    # data (as pandas dataframes)
    X = secondary_mushroom.data.features
    y = secondary_mushroom.data.targets

    # print metadata of the variables
    print(secondary_mushroom.variables)

    # p = poisonous, e = edible
    count_p = (y['class'] == 'p').sum()
    count_e = (y['class'] == 'e').sum()

    print(f"Count of poisonous samples: {count_p}, Count of edible samples: {count_e}")

    # Count the number of missing values per feature
    missing_counts = X.isna().sum()

    # Calculate the percentage of missing values per feature
    missing_percentages = (missing_counts / len(X)) * 100
    print(f"Missing percentages: {missing_percentages}")

    # Plot the percentages
    plt.figure(figsize=(10, 6))
    missing_percentages.plot(kind='bar', color='gray', edgecolor='black')
    plt.title("Percentage of Missing Values Per Feature", fontsize=16)
    plt.xlabel("Features", fontsize=14)
    plt.ylabel("Percentage of Missing Values (%)", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(np.arange(0, 100, step=5))
    plt.tight_layout()
    plt.show()


def handle_missing_data_manually(X):
    """
        Imputation of the features with high missing rate could lead to a considerable bias
        in the results. Since ID3 doesn't handle missing values we have to handle this obstacle before 
        performing the algorithm.
    """
    feature_names_to_impute = ["cap-surface", "gill-attachment", "ring-type", "gill-spacing"]
    feature_names_to_be_removed = ["stem-surface", "stem-root", "veil-type", "veil-color", "spore-print-color"]
    X_imputed = impute_categorical_mode(X, feature_names_to_impute)
    X_imputed = X_imputed.drop(columns=feature_names_to_be_removed, inplace=False)
    return X_imputed


def zero_one_loss(y_true, y_pred):
    """Calculate 0-1 loss (misclassification rate)."""
    return np.mean(y_true != y_pred)


def train_and_evaluate_id3(X_train, X_test, y_train, y_test, max_depth, min_samples_split, criterion):
    """Train ID3 and compute train/test error for given parameters."""
    model = ID3(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        missing_penalty_power=2,
        min_impurity_decrease=0,
        criterion=criterion
    )

    # Train the model
    model.fit(X_train, y_train)

    # Predict train and test sets
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Compute 0-1 loss (misclassification rate)
    train_error = zero_one_loss(y_train, y_train_pred)
    test_error = zero_one_loss(y_test, y_test_pred)

    return max_depth, min_samples_split, train_error, test_error


def evaluate_model_and_plot_results(X, y, criterion="information_gain", plot_title=None):
    # Split the dataset into train and test sets while maintaining class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)

    # Define hyperparameter search space
    max_depth_values = np.arange(2, 32, 4)
    min_samples_split_values = [3000, 1000, 500, 100, 50, 10, 2]

    # Generate all parameter combinations
    param_combinations = [(d, s) for d in max_depth_values for s in min_samples_split_values]

    print(f"Starting grid search over {len(param_combinations)} iterations...")

    # Run evaluations in parallel
    results = Parallel(n_jobs=-1)(
        delayed(train_and_evaluate_id3)(X_train, X_test, y_train, y_test, d, s, criterion)
        for d, s in tqdm(param_combinations, desc="Processing")
    )

    # Convert results to array for easy processing
    results = np.array(results)

    # Extract values for plotting
    max_depth_values = results[:, 0]
    min_samples_split_values = results[:, 1]
    train_errors = results[:, 2]
    test_errors = results[:, 3]

    # Create a combined x-axis label (max_depth, min_samples_split)
    labels = [f"d={int(d)}, s={int(s)}" for d, s in zip(max_depth_values, min_samples_split_values)]

    # Reduce label density (show every nth label)

    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(labels, train_errors, marker="o", label="Train Error", linestyle="-", color="blue")
    plt.plot(labels, test_errors, marker="s", linestyle="dashed", label="Test Error", color="red")

    plt.xlabel("Max Depth and Min Samples Split (d, s)")
    plt.ylabel("Error (0-1 Loss)")
    plt.title(f"Train and Test Error vs Max Depth & Min Samples Split|{plot_title}")

    # Adjust x-axis labels
    plt.xticks(rotation=45, ha="right")
    # plt.xlim(0, len(labels))
    # plt.tick_params(axis="x", which="major", labelsize=8)

    # Increase bottom margin to avoid label cutoff
    plt.subplots_adjust(bottom=0.2)
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()

    print("Grid search completed successfully.")


def perform_random_forest(X,
                          y,
                          n_estimators=10,
                          max_features="sqrt",
                          max_depth=5,
                          min_samples_split=2,
                          missing_penalty_power=1.5):
    # Split the dataset into train and test sets while maintaining class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,
                                                        random_state=random_state)
    """
    RandomForest training 
    """
    rf = RandomForest(
        base_estimator_class=ID3,
        n_estimators=n_estimators,
        max_features=max_features,
        random_state=random_state,
        # ID3-specific params:
        stopping_criteria=None,
        max_depth=max_depth,
        missing_penalty_power=missing_penalty_power,
        min_samples_split=min_samples_split,
        min_impurity_decrease=0.0,
    )

    rf.fit(X_train, y_train)
    print("RF Accuracy:", rf.score(X_test, y_test))


def visualize_trees(X, y, max_depth=5, min_samples_split=2, missing_penalty_power=2):
    # Extracting feature names
    feature_names = X.columns.tolist()
    X = X.to_numpy()
    # Split the dataset into train and test sets while maintaining class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    """
    Visualization of default ID3 with information Gain
    """
    model_information_gain = ID3(max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 missing_penalty_power=missing_penalty_power,
                                 min_impurity_decrease=0,
                                 )
    model_information_gain.fit(X_train, y_train)
    information_gain_accuracy = model_information_gain.score(X_test, y_test)
    print("Information gain accuracy:", information_gain_accuracy)
    model_information_gain.visualize_tree(feature_names=feature_names, file_name="tree_information_gain")
    """
    Visualization of ID3 with Gain Ratio gain
    """
    model_gain_ratio = ID3(max_depth=max_depth,
                           min_samples_split=min_samples_split,
                           missing_penalty_power=missing_penalty_power,
                           min_impurity_decrease=0,
                           criterion="gain_ratio"
                           )
    model_gain_ratio.fit(X_train, y_train)
    information_gain_accuracy = model_gain_ratio.score(X_test, y_test)
    print("Gain Ratio accuracy:", information_gain_accuracy)
    model_gain_ratio.visualize_tree(feature_names=feature_names, file_name="tree_gain_ratio")
    """
    Visualization of ID3 with Gini Index 
    """
    model_gini = ID3(max_depth=max_depth,
                           min_samples_split=min_samples_split,
                           missing_penalty_power=missing_penalty_power,
                           min_impurity_decrease=0,
                           criterion="gini"
                           )
    model_gini.fit(X_train, y_train)
    information_gain_accuracy = model_gini.score(X_test, y_test)
    print("Gini accuracy:", information_gain_accuracy)
    model_gini.visualize_tree(feature_names=feature_names, file_name="tree_gini")


def visualize_tree_pruning(X, y, max_depth=5, min_samples_split=2, missing_penalty_power=2, alpha=0, criterion="information_gain"):
    # Extracting feature names
    feature_names = X.columns.tolist()
    X = X.to_numpy()
    # Split the dataset into train and test sets while maintaining class balance
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=random_state)
    """
    Visualization of default ID3 with information Gain
    """
    model = ID3(max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 missing_penalty_power=missing_penalty_power,
                                 min_impurity_decrease=0,
                                 criterion=criterion
                                 )
    model.fit(X_train, y_train)
    model.visualize_tree(feature_names=feature_names, file_name="tree_information_gain_not_pruned")
    print("Pruning...")
    model.cost_complexity_prune(X_test, y_test, alpha=alpha)
    model.visualize_tree(feature_names=feature_names, file_name="tree_information_gain_pruned")


def perform_hyperparameter_tuning(X, y):
    # Defining the range of the hyperparameters
    search_space = {
        'max_depth': Categorical(np.arange(2, 32, 4)),
        'min_samples_split': Categorical([3000, 1000, 500, 100, 50, 10, 2]),
        'missing_penalty_power': Categorical([0.5, 1.0, 1.5, 2.0]),  # Specific values
        'criterion': Categorical(["information_gain", "gain_ratio", "gini"])
    }

    # Set up cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    # Instantiate Bayesian optimization with cross-validation
    bayes_search = BayesSearchCV(
        ID3(),  # Your ID3 model
        search_spaces=search_space,
        n_iter=30,  # Number of iterations for Bayesian search
        scoring='accuracy',  # Scoring metric
        cv=cv,
        n_jobs=-1,  # Use all CPU cores
        random_state=random_state
    )

    # Fit Bayesian optimizer
    bayes_search.fit(X, y)

    # Print best parameters
    print("Best hyperparameters found:")
    print(bayes_search.best_params_)

    # Print best score
    print("Best accuracy achieved:", bayes_search.best_score_)


def main():
    # fetch dataset
    secondary_mushroom = fetch_ucirepo(id=848)
    # data (as pandas dataframes)
    X = secondary_mushroom.data.features
    y = secondary_mushroom.data.targets.to_numpy().ravel()
    """ 
       Encoding labels from string values to integers 
       edible (e) => 0 , poisonous (p) p=>1
       """
    y = np.where(y == 'e', 0, 1)

    while True:
        print("\nSelect an option:")
        print("1. Visualize dataset")
        print("2. Evaluate model and plot results")
        print("3. Visualize decision trees")
        print("4. Visualize pruned decision trees")
        print("5. Perform hyperparameter tuning")
        print("6. Perform Random Forest classification")
        print("7. Exit")

        choice = input("Enter your choice: ")

        if choice == '1':
            visualize_dataset()
        elif choice == '2':
            criterion = input("Enter criterion (information_gain, gain_ratio, gini): ")
            evaluate_model_and_plot_results(X.to_numpy(), y, criterion=criterion, plot_title=criterion)
        elif choice == '3':
            max_depth = int(input("Enter max depth (default=5): ") or 5)
            min_samples_split = int(input("Enter min samples split (default=2): ") or 2)
            missing_penalty_power = float(input("Enter missing penalty power (default=2): ") or 2)
            visualize_trees(X, y, max_depth, min_samples_split, missing_penalty_power)
        elif choice == '4':
            max_depth = int(input("Enter max depth (default=5): ") or 5)
            min_samples_split = int(input("Enter min samples split (default=2): ") or 2)
            missing_penalty_power = float(input("Enter missing penalty power (default=2): ") or 2)
            alpha = float(input("Enter alpha for pruning (default=0): ") or 0)
            criterion = input(
                "Enter criterion (information_gain, gain_ratio, gini, default=information_gain): ") or "information_gain"
            visualize_tree_pruning(X, y, max_depth, min_samples_split, missing_penalty_power, alpha, criterion)
        elif choice == '5':
            perform_hyperparameter_tuning(X.to_numpy(), y)
        elif choice == '6':
            n_estimators = int(input("Enter number of estimators (default=10): ") or 10)
            max_features = input("Enter max features (default=sqrt): ") or "sqrt"
            max_depth = int(input("Enter max depth (default=5): ") or 5)
            min_samples_split = int(input("Enter min samples split (default=2): ") or 2)
            missing_penalty_power = float(input("Enter missing penalty power (default=1.5): ") or 1.5)
            perform_random_forest(X.to_numpy(), y, n_estimators=n_estimators, max_features=max_features,
                                  max_depth=max_depth, min_samples_split=min_samples_split,
                                  missing_penalty_power=missing_penalty_power)
        elif choice == '7':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    main()
