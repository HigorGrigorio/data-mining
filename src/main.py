import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.mappers import Mapper

CORRELATION_MATRIX_SIZE = 20


# function data load
def load_data(
    file_path: str,
    sep: str = ";",
    header: int = 0,
    index_col: bool = False,
    na_values: str = "unknown",
    usecols: list = None,
) -> pd.DataFrame:
    return pd.read_csv(
        file_path,
        sep=sep,
        header=header,
        index_col=index_col,
        na_values=np.nan,
        usecols=usecols,
        converters={
            "y": lambda x: Mapper.result_status(x),
            "job": lambda x: Mapper.index_job(x) if x != na_values else np.nan,
            "marital": lambda x: Mapper.marital_status(x) if x != na_values else np.nan,
            "education": lambda x: Mapper.education_level(x) if x != na_values else np.nan,
            "default": lambda x: Mapper.default_status(x) if x != na_values else np.nan,
            "housing": lambda x: Mapper.housing_loan(x) if x != na_values else np.nan,
            "loan": lambda x: Mapper.loan_status(x) if x != na_values else np.nan,
            "month": lambda x: Mapper.month_index(x),
            "day_of_week": lambda x: Mapper.day_of_week_index(x),
            "contact": lambda x: Mapper.contact_type(x),
            "poutcome": lambda x: Mapper.poutcome_status(x),
        },
    )


def data_summary(data: pd.DataFrame) -> None:
    print(f"Data examples:\n{data.head(10)}")
    # print(f"Data info:\n{data.info()}")
    print(f"Data desc:\n{data.describe()}")
    print(f"Data shape:\n{data.shape}")
    # print(f"Data columns:\n{data.columns}")
    print(f"Data missing values:\n{data.isnull().sum()}")


def data_preprocessing(data: pd.DataFrame):
    data = data.dropna()
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


def main():
    # data load
    cols = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "housing",
        "loan",
        "contact",
        "month",
        "day_of_week",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "emp.var.rate",
        "cons.price.idx",
        "cons.conf.idx",
        "euribor3m",
        "nr.employed",
        "y",
    ]
    data_path = __file__.replace(
        "main.py", "data/bank-additional/bank-additional/bank-additional-full.csv"
    )
    data = load_data(
        data_path,
        usecols=cols,
    )
    data_summary(data)
    data = data_preprocessing(data)
    data_summary(data)

    mrx = correlation_matrix(data)

    plot_correlation_matrix(mrx)


if __name__ == "__main__":
    main()
