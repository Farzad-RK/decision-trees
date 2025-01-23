import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
import numpy as np


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


def main():
    visualize_dataset()


if __name__ == '__main__':
    main()
