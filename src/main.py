import itertools
import json
from io import StringIO
from math import ceil
from turtle import width
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyparsing import line
import seaborn as sns
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
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
            data = data.dropna()  # drop rows with missing values
        case "mean":
            data = data.fillna(
                data.mean()
            )  # fill missing values with the mean of the column
        case "mode":
            data = data.fillna(
                data.mode().iloc[0]
            )  # fill missing values with the mode of the column
        case "median":
            data = data.fillna(
                data.median()
            )  # fill missing values with the median of the column
        case _:
            raise ValueError("Invalid method")
    return data


def correlation_matrix(data: pd.DataFrame):
    corr_matrix = data.corr("pearson")
    print(f"Correlation matrix:\n{corr_matrix}")
    return corr_matrix


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    size: tuple = (CORRELATION_MATRIX_SIZE, CORRELATION_MATRIX_SIZE),
):
    """
    Plot correlation matrix with heatmap
    """
    plt.figure(figsize=size)
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix Heatmap")
    plt.show()


def show_information_data_frame(
    data: pd.DataFrame,
    correl_matrix: bool = False,
    size: tuple = (CORRELATION_MATRIX_SIZE, CORRELATION_MATRIX_SIZE),
):
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
        plot_correlation_matrix(mrx, size=size)


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


def pca3d(x_score, df: pd.DataFrame, targetColumn):
    pca = PCA()
    principalComponents = pca.fit_transform(x_score)  # fit the data and transform it
    print("Explained variance per component:")
    print(pca.explained_variance_ratio_.tolist())
    print("\n\n")

    principalDf = pd.DataFrame(
        data=principalComponents[:, 0:3],
        columns=[
            "principal component 1",
            "principal component 2",
            "principal component 3",
        ],
    )
    finalDf = pd.concat([principalDf, df[[targetColumn]]], axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("Principal Component 1", fontsize=15)
    ax.set_ylabel("Principal Component 2", fontsize=15)
    ax.set_zlabel("Principal Component 3", fontsize=15)
    ax.set_title("3 component PCA", fontsize=20)
    targets = {0: "no", 1: "yes"}
    colors = ["r", "g"]
    for target, color in zip(targets, colors):
        indicesToKeep = finalDf[targetColumn] == target
        ax.scatter(
            finalDf.loc[indicesToKeep, "principal component 1"],
            finalDf.loc[indicesToKeep, "principal component 2"],
            finalDf.loc[indicesToKeep, "principal component 3"],
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
    VisualizePcaProjection(finalDf, target)


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


def mix_cols(df: pd.DataFrame, col1: str, col2: str, target: str, plot: bool = True):
    """
    Mix two columns and plot the result

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        col1 (str): The first column to be mixed.
        col2 (str): The second column to be mixed.
        target (str): The target column.

    Returns:
        None
    """
    # remove the target column
    targ = df[target].values
    df.drop([target], axis=1, inplace=True)

    # mix the columns
    df[str(col1 + "_" + col2)] = (df[col1] + df[col2]) / 2
    df.drop([col1, col2], axis=1, inplace=True)

    # add the target column back
    df[target] = targ

    # plot the mixed column
    if plot:
        show_information_data_frame(
            df,
            correl_matrix=True,
            size=(CORRELATION_MATRIX_SIZE, CORRELATION_MATRIX_SIZE),
        )
    return df


def balance(df: pd.DataFrame):
    return (
        df.groupby("y")
        .apply(lambda x: x.sample(n=3500, replace=False))
        .reset_index(drop=True)
    )


def k_means(
    df: pd.DataFrame, n_clusters: int, target: str = "y", features: list[str] = []
):
    """
    Perform K-means clustering on the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame containing the data.
        n_clusters (int): The number of clusters to form.

    Returns:
        None
    """

    # if no features are provided, use all columns except the target column
    if not features:
        features = df.columns.tolist()
        features.remove(target)

    # drop rows with missing values
    df.dropna(inplace=True)

    # create train and test data
    x_train, x_test, y_train, y_test = train_test_split(
        df[features], df[target], test_size=0.5, random_state=0
    )

    # normalize the data
    x_train = preprocessing.normalize(x_train)
    x_test = preprocessing.normalize(x_test)

    pca = PCA()
    principalComponents = pca.fit_transform(x_train)
    testComponents = pca.fit_transform(x_test)

    # perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(principalComponents[:, 0:2])

    # calculate the silhouette score
    score = silhouette_score(testComponents[:, 0:2], kmeans.labels_, metric="euclidean")

    return kmeans, score


def visualize_decision_tree(clf, features):
    plt.figure(figsize=(20, 20))
    tree.plot_tree(clf, feature_names=features, class_names=["no", "yes"], filled=True)
    plt.show()


def decision_tree(x, y, visualize=False):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=0, stratify=y
    )

    x_train = preprocessing.normalize(x_train, "max")
    x_test = preprocessing.normalize(x_test, "max")
    clf = DecisionTreeClassifier(random_state=0, max_leaf_nodes=5)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(f"Accuracy: {round(clf.score(x_test, y_test)*100,2)}%")

    if visualize:
        cm = confusion_matrix(y_test, y_pred)
        plot_confusion_matrix(
            cm, ["no", "yes"], normalize=False, title="Confusion matrix DT"
        )
        plot_confusion_matrix(
            cm, ["no", "yes"], normalize=True, title="Confusion matrix DT Normalized"
        )
        visualize_decision_tree(clf, x)


def euclidian_distance(row1, row2) -> float:
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return distance ** (0.5)


def get_neighbors(train, test_row, num_neighbors: int):
    distances = []
    for train_row in train:
        dist = euclidian_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = []
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


def knn_predict(train, test_row, num_neighbors):
    neighbors = get_neighbors(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def knn_manual(x, y, n: int):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=0, stratify=y
    )

    x_train = preprocessing.normalize(x_train, "max")
    x_test = preprocessing.normalize(x_test, "max")

    train = np.concatenate((x_train, y_train.reshape(-1, 1)), axis=1)
    test = np.concatenate((x_test, y_test.reshape(-1, 1)), axis=1)

    predictions = []

    for row in test:
        output = knn_predict(train, row, n)
        predictions.append(output)

    cm = confusion_matrix(y_test, predictions)
    plot_confusion_matrix(
        cm, ["no", "yes"], normalize=False, title="Confusion matrix KNN"
    )
    plot_confusion_matrix(
        cm, ["no", "yes"], normalize=True, title="Confusion matrix KNN Normalized"
    )

    return predictions


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def knn_sklearn(x, y):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, random_state=0, stratify=y
    )
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(x_train, y_train)

    y_pred = knn.predict(x_test)
    print(f"Accuracy: {round(knn.score(x_test, y_test)*100,2)}%")

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, ["no", "yes"], normalize=False, title="Confusion matrix KNN"
    )
    plot_confusion_matrix(
        cm, ["no", "yes"], normalize=True, title="Confusion matrix KNN Normalized"
    )
    plt.show()


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
    df = pd.DataFrame(norm.normalize(df, "sklearn-min-max"), columns=features + [target])
    df_balanced = balance(df.copy())
    df_balanced.hist(bins=50, figsize=(20, 15))
    plt.show()

    # getting the features to be used in the analysis
    x = df_balanced.loc[:, features].values

    # getting the target to be predicted
    y = df_balanced.loc[:, target].values

    # fit x using z score normalization
    # x_score = norm.normalize(x, "sklearn-min-max")
    # using normalized columns to create a new dataframe
    # normalized_df = pd.DataFrame(x_score, columns=features)
    # normalized_df = pd.concat(
    # [normalized_df, df[target]], axis=1  # axis = 0 for rows, axis = 1 for columns
    # )

    # show_information_data_frame(df, correl_matrix=False)
    # show_information_data_frame(normalized_df, correl_matrix=True)
    # show_information_data_frame(normalized_df, correl_matrix=False)

    # PCA projection
    # freq = normalized_df.copy()
    # freq["freq"] = 1.0 / freq.groupby(target)[target].transform("count")
    # pca(x_score, normalized_df.sample(n=4500, weights=freq.freq), target=target)
    # pca3d(
    #     x_score,
    #     normalized_df.sample(n=4500, weights=freq.freq),
    #     target,
    # )

    # Plot the dda of cols
    # col_dda(normalized_df.sample(n=4500, weights=freq.freq).copy(), "age", "y", "bar", bins=5)
    # col_dda(normalized_df.sample(n=4500, weights=freq.freq).copy(), "age", "y", "line-percentage", bins=5)
    # col_dda(df.copy(), "job", "y", "pie")
    # col_dda(df.copy(), "education", "y", "bar")
    # col_dda(df.copy(), "education", "y", "line-percentage")
    # col_dda(df.copy(), "marital", "y", "bar")
    # col_dda(df.copy(), "marital", "y", "line-percentage")
    # col_dda(df.copy(), "loan", "y", "bar")
    # col_dda(df.copy(), "housing", "y", "bar")

    # Mix two columns
    # mix_df = mix_cols(df.copy(), "emp.var.rate", "euribor3m", target, plot=False)
    # mix_df = mix_cols(
    # mix_df.copy(), "emp.var.rate_euribor3m", "nr.employed", target, plot=False
    # )

    # show_information_data_frame(mix_df, correl_matrix=True)

    # K-means clustering
    # K = range(2, 10)
    # fits = []
    # scores = []
    # for k in K:
    #     fit, score = k_means(df_balanced.copy(), k, target=target, features=features)
    #     fits.append(fit)
    #     scores.append(score)

    # plt.figure(figsize=(10, 5))
    # sns.lineplot(x=K, y=scores)
    # plt.xlabel("Number of clusters")
    # plt.ylabel("Silhouette score")
    # plt.title("Silhouette score vs Number of clusters")
    # plt.show()

    # Decision tree
    # decision_tree(x, y, True)

    # KNN - Manual implementation
    # predictions = knn_manual(x, y, 5)

    # KNN - Sklearn implementation
    # knn_sklearn(x, y)


if __name__ == "__main__":
    main()
