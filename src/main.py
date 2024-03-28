import json
from io import StringIO
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

import normalization as norm
from mappers import BaseMapper

CORRELATION_MATRIX_SIZE = 20


def data_preprocessing(data: pd.DataFrame, method: str):
    """
    Preprocesses the given DataFrame based on the specified method.

    Parameters:
        data (pd.DataFrame): The input DataFrame to be preprocessed.
        method (str): The method to be used for preprocessing. Valid options are "clean", "mean", "mode", and "median".

    Returns:
        pd.DataFrame: The preprocessed DataFrame.

    Raises:
        ValueError: If an invalid method is provided.
    """
    match method:
        case "clean":
            data = data.dropna()
        case "mean":
            data = data.fillna(data.mean())
        case "mode":
            data = data.fillna(data.mode().iloc[0])
        case "median":
            data = data.fillna(data.median())
        case _:
            raise ValueError("Invalid method")
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


def VisualizePcaProjection(finalDf, targetColumn):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_title("2 component PCA", fontsize=20)
    targets = {0: "no", 1: "yes"}
    colors = ["r", "g"]
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
            c=color,
            s=50,
        )
    ax.legend(targets.values())
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


def col_dda(df: pd.DataFrame, col: str | list[str], target: str, graph: str, **kwargs):
    """
    This function receives a dataframe, a column name, a target column name and a graph type.
    It plots the distribution of the column values and the target values.
    """
    match graph:
        case "hist":
            if df[col].dtype == "object":
                df_copy = BaseMapper.get_mapper(col).revert_list(df[col])
                df[col] = df_copy

            df[col].hist()
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        case "box":
            df.boxplot(column=col, by=target)
            plt.title(f"Distribution of {col} by {target}")
            plt.xlabel(target)
            plt.ylabel(col)
            plt.show()
        case "bar":
            if kwargs.get("bins"):
                df[col].value_counts(bins=kwargs.get("bins"), sort=False).plot(
                    kind="bar", rot=0
                )
            else:
                df[col].value_counts().plot(kind="bar")
            plt.title(f"Bar plot of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        case "pie":
            df_copy = BaseMapper.get_mapper(col).revert_list(df[col])
            df[col] = df_copy
            df[col].value_counts().plot(kind="pie", autopct="%1.1f%%")
            plt.title(f"Pie plot of {col}")
            plt.ylabel(col)
            plt.show()
        case "density":
            df[col].value_counts(sort=False).plot(kind="density")
            plt.title(f"Count plot of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        case "scatter":
            df_copy = BaseMapper.get_mapper(col).revert_list(df[col])
            df[col] = df_copy
            plt.scatter(df[col], df[target])
            plt.title(f"Scatter plot of {col} and {target}")
            plt.xlabel(col)
            plt.ylabel(target)
            plt.show()
        case _:  # default case
            raise ValueError("Invalid graph type")


INPUT_FILE = "./src/data/bank-additional/bank-additional/bank-additional-full.csv"
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
    io = StringIO(open("./src/data/entry.json", "r").read())
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
        converters={
            col: _make_map_adapter(col) for col in categorical
        },  # the converters
    )

    # perform data preprocessing
    df = data_preprocessing(df, "mode")

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

    # show_information_data_frame(df, correl_matrix=False)
    # show_information_data_frame(normalized_df, correl_matrix=True)
    show_information_data_frame(normalized_df, correl_matrix=False)

    # PCA projection
    # pca(x_score, df, target)

    # Plot the dda of cols
    # col_dda(df, "age", "y", "bar", bins=ceil(df["age"].max() / 10))
    # col_dda(df, "job", "y", "pie")
    # col_dda(df, "education", "y", "pie")
    # col_dda(df, "marital", "y", "pie")
    col_dda(df, "marital", "y", "box")


if __name__ == "__main__":
    main()
