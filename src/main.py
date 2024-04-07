from enum import auto
import json
from io import StringIO
from math import ceil
from turtle import width

import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import line
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
            data = data.dropna() # drop rows with missing values
        case "mean":
            data = data.fillna(data.mean()) # fill missing values with the mean of the column
        case "mode":
            data = data.fillna(data.mode().iloc[0]) # fill missing values with the mode of the column
        case "median":
            data = data.fillna(data.median()) # fill missing values with the median of the column
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


def _is_categorical(col: str) -> bool:
    io = StringIO(open("./src/data/entry.json", "r").read())
    entry = json.load(io)
    if col in entry["categorical"]:
        return True
    return False


def col_dda(df: pd.DataFrame, col: str | list[str], target: str, graph: str, **kwargs):
    """
    Plot various types of graphs based on the column values and the target values.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col (str | list[str]): The column(s) to be plotted.
        target (str): The target column.
        graph (str): The type of graph to be plotted. Options: "bar", "box", "pie", "density", "scatter", "kde".
        **kwargs: Additional keyword arguments for customizing the plots.

    Raises:
        ValueError: If an invalid graph type is provided.

    Returns:
        None
    """
    match graph:
        case "bar":
            # Plot the relative frequency of the column values by the target values
            if _is_categorical(col):
                df_copy = BaseMapper.get_mapper(col).revert_list(df[col])
                df[col] = df_copy
            df_copy = BaseMapper.get_mapper(target).revert_list(df[target])
            df[target] = df_copy
            if kwargs.get("bins"):
                df.groupby(pd.cut(df[col], bins=kwargs.get("bins")), observed=False)[
                    target
                ].value_counts(sort=False).unstack().plot(kind="bar", stacked=False)
            else:
                df.groupby(col, observed=False)[target].value_counts(
                    sort=False
                ).unstack().plot(kind="bar", stacked=False)
            plt.title(f"Bar plot of {col} by {target}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()
        case "line-percentage":
            # percentage of yes and no, for each category
            if _is_categorical(col):
                df_copy = BaseMapper.get_mapper(col).revert_list(df[col])
                df[col] = df_copy
            df_copy = BaseMapper.get_mapper(target).revert_list(df[target])
            df[target] = df_copy
            if kwargs.get("bins"):
                df = (
                    df.groupby(
                        pd.cut(df[col], bins=kwargs.get("bins")), observed=False
                    )[target]
                    .value_counts(sort=False)
                    .unstack()
                )
            else:
                df = (
                    df.groupby(col, observed=False)[target]
                    .value_counts(sort=False)
                    .unstack()
                )
            df = df.div(df.sum(axis=1), axis=0) * 100
            print(f"Percentage of yes and no for each category:\n{df}")
            df.plot(kind="line", stacked=False, xticks=range(len(df.index)))
            plt.title(f"Line plot of {col} by {target}")
            plt.xlabel(col)
            plt.ylabel("Percentage")
            plt.show()
        case "box":
            # Plot the boxplot of the column values by the target values
            df.boxplot(
                column=col,
                by=target,
            )
            plt.ylabel(col)
            plt.show()
        case "pie":
            # Plot the pie chart of the column values by the target values
            if _is_categorical(col):
                df_copy = BaseMapper.get_mapper(col).revert_list(df[col])
                df[col] = df_copy
            df_copy = BaseMapper.get_mapper(target).revert_list(df[target])
            df[target] = df_copy
            df.groupby(col)[target].value_counts().unstack().plot(
                kind="pie",
                subplots=True,
                figsize=(15, 15),
                autopct="%1.1f%%",
                title=f"Pie plot of {col} by {target}",
            )
            plt.show()
        case "density":
            # Plot the density plot of the column values by the target values
            df.groupby(col)[target].plot(kind="density")
            plt.title(f"Density plot of {col} by {target}")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.show()
        case "scatter":
            # Plot the scatter plot of the column values by the target values
            if _is_categorical(col):
                df_copy = BaseMapper.get_mapper(col).revert_list(df[col])
                df[col] = df_copy
            df_copy = BaseMapper.get_mapper(target).revert_list(df[target])
            df[target] = df_copy
            df.plot.scatter(x=col, y=target)
            plt.title(f"Scatter plot of {col} by {target}")
            plt.xlabel(col)
            plt.ylabel(target)
            plt.show()
        case "kde":
            # Plot the kernel density estimation plot of the column values by the target values
            df.groupby(col)[target].plot(kind="kde")
            plt.title(f"KDE plot of {col} by {target}")
            plt.xlabel(col)
            plt.ylabel("Density")
            plt.show()
        case "line":
            # Plot the line plot of the column values by the target values
            df.groupby(col, sort=False)[target].plot(kind="line")
            plt.title(f"Line plot of {target} by {col}")
            plt.xlabel(target)
            plt.ylabel(col)
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
    df = data_preprocessing(df, "clean")

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
    # show_information_data_frame(normalized_df, correl_matrix=False)

    # PCA projection
    # pca(x_score, normalized_df, target)

    # Plot the dda of cols
    # col_dda(df.copy(), "age", "y", "bar", bins=ceil(df["age"].max() / 10))
    # col_dda(df.copy(), "age", "y", "line-percentage", bins=ceil(df["age"].max() / 10))
    # col_dda(df.copy(), "job", "y", "pie")
    # col_dda(df.copy(), "education", "y", "bar")
    # col_dda(df.copy(), "education", "y", "line-percentage")
    # col_dda(df.copy(), "marital", "y", "bar")
    # col_dda(df.copy(), "marital", "y", "line-percentage")
    col_dda(df.copy(), "loan", "y", "bar")
    col_dda(df.copy(), "housing", "y", "bar")


if __name__ == "__main__":
    main()
