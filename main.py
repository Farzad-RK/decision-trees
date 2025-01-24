import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np
from utils import impute_categorical_mode
from id3_tree import ID3


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


def handle_missing_data(X):
    """
        feature_name            missing percentage
        cap-surface             23.121387
        gill-attachment         16.184971
        ring-type                4.046243
        gill-spacing            41.040462
        stem-surface            62.427746
        stem-root               84.393064
        veil-type               94.797688
        veil-color              87.861272
        spore-print-color       89.595376
        
        Imputation of the features with high missing rate could lead to a considerable bias
        in the results. Since ID3 doesn't handle missing values we have to handle this obstacle before 
        performing the algorithm.
    """
    feature_names_to_impute = ["cap-surface", "gill-attachment", "ring-type", "gill-spacing", "stem-surface"]
    feature_names_to_be_removed = ["stem-root", "veil-type", "veil-color", "spore-print-color"]
    X_imputed = impute_categorical_mode(X, feature_names_to_impute)
    X_imputed = X_imputed.drop(columns=feature_names_to_be_removed, inplace=False)
    return X_imputed


def perform_id3_classification(dataset):
    # data (as pandas dataframes)
    X = dataset.data.features
    X_imputed = handle_missing_data(X).to_numpy()
    y = dataset.data.targets.to_numpy().ravel()

    """ 
    Encoding labels from string values to integers 
    edible (e) => 0 , poisonous (p) p=>1
    """
    y = np.where(y == 'e', 0, 1)

    # Train ID3 Decision Tree
    id3_classifier = ID3(stopping_criteria={"max_depth": 1})
    train_loss, test_loss = id3_classifier.cross_validate(X_imputed, y, 2)
    print(train_loss, test_loss)


def main():
    # fetch dataset
    secondary_mushroom = fetch_ucirepo(id=848)
    # #visualize_dataset(secondary_mushroom)
    perform_id3_classification(secondary_mushroom)


if __name__ == '__main__':
    main()
