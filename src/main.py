import json
from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

import normalization as norm
from mappers import BaseMapper

CORRELATION_MATRIX_SIZE = 20


def data_preprocessing(data: pd.DataFrame):
    data = data.dropna() # drop missing values
    return data


def correlation_matrix(data: pd.DataFrame):
    corr_matrix = data.corr("pearson")
    print(f"Correlation matrix:\n{corr_matrix}")
    return corr_matrix


def plot_correlation_matrix(corr_matrix: pd.DataFrame):
    """
    Plot correlation matrix with heatmap
    """
    plt.figure(figsize=(CORRELATION_MATRIX_SIZE, CORRELATION_MATRIX_SIZE))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def show_information_data_frame(data: pd.DataFrame, correl_matrix: bool = False):
    """
    Show information about the data frame
    """
    print(f"Data examples:\n{data.head(10)}")
    print(f"Data info:\n{data.info()}")
    print(f"Data desc:\n{data.describe()}")
    print(f"Data shape:\n{data.shape}")
    print(f"Data columns:\n{data.columns}")
    print(f"Data missing values:\n{data.isnull().sum()}")

    if correl_matrix:
        mrx = correlation_matrix(data)
        plot_correlation_matrix(mrx)


def VisualizePcaProjection(df, targetColumn):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)
    targets = [
        0,
        1,
    ]
    colors = ["r", "g"]
    for target, color in zip(targets, colors):
        indicesToKeep = df[targetColumn] == target
        ax.scatter(
            df.loc[indicesToKeep, "principal component 1"],
            df.loc[indicesToKeep, "principal component 2"],
            c=color,
            s=50,
        )
    ax.legend(targets)
    ax.grid()
    plt.show()


def pca(x_score, df: pd.DataFrame, target):
    pca = PCA()
    principalComponents = pca.fit_transform(x_score)  # fit the data and transform it
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")

    principalDf = pd.DataFrame(
        data=principalComponents[:, 0:2],
        columns=["principal component 1", "principal component 2"],
    )
    finalDf = pd.concat([principalDf, df[[target]]], axis=1)
    VisualizePcaProjection(finalDf, "y")


def col_dda(df: pd.DataFrame, col: str, target: str, graph: str):
    """
    This function receives a dataframe, a column name, a target column name and a graph type.
    It plots the distribution of the column values and the target values.
    """
    if graph == "hist":

        df[col].hist()
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.show()
    elif graph == "box":
        df.boxplot(column=col, by=target)
        plt.title(f"Distribution of {col} by {target}")
        plt.xlabel(target)
        plt.ylabel(col)
        plt.show()
    else:
        raise ValueError("Invalid graph type")


INPUT_FILE = "./data/bank-additional/bank-additional/bank-additional-full.csv"
NA_VALUE = "unknown"


def _make_map_adapter(column: str):
    return BaseMapper.get_mapper(column).map


def main():
    """
    This is the main function that performs the analysis on the data.

    It loads the entry columns from data/entry.json, reads the data from a CSV file,
    performs data preprocessing, normalization, and displays information about the data.

    Args:
        None

    Returns:
        None
    """

    # load the entry columns from data/entry.json
    io = StringIO(open("./data/entry.json", "r").read())
    entry = json.load(io)

    # the columns to be used in the analysis
    names = entry["names"]

    # the features are the columns to be used in the analysis
    features = entry["features"]

    # the target is the column to be predicted, it should not be in the features,
    # and it should be a list with only one element
    target = entry["target"]

    # the categorical columns are the columns that can be mapped to integers
    categorical = entry["categorical"]

    # load the data from the file
    df = pd.read_csv(
        INPUT_FILE,  # the path to the data file
        header=0,  # the header of the file
        index_col=False,  # the index column
        na_values=NA_VALUE,  # the missing values
        sep=";",  # the separator of the file
        usecols=names,  # the columns to be used in the analysis
        converters={col: _make_map_adapter(col) for col in categorical},  # the converters
    )

    # perform data preprocessing - drop missing values
    df = data_preprocessing(df)

    # getting the features to be used in the analysis
    x = df.loc[:, features].values

    # getting the target to be predicted
    y = df.loc[:, target].values

    # fit x using z score normalization
    x_score = norm.normalize(x, "sklearn-z-score")
    # using normalized columns to create a new dataframe
    normalized_df = pd.DataFrame(x_score, columns=features)
    normalized_df = pd.concat(
        [normalized_df, df[target]], axis=1  # axis = 0 for rows, axis = 1 for columns
    )

    # show_information_data_frame(normalized_df, correl_matrix=True)

    # PCA projection
    # pca(x_score, df, target)

    # Plot the dda of cols
    col_dda(df, "marital", "y", "hist")


if __name__ == "__main__":
    main()
