import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

CORRELATION_MATRIX_SIZE = 20

def index_job(job: str) -> int:
    jobs = {
        "admin.": 0,
        "blue-collar": 1,
        "entrepreneur": 2,
        "housemaid": 3,
        "management": 4,
        "retired": 5,
        "self-employed": 6,
        "services": 7,
        "student": 8,
        "technician": 9,
        "unemployed": 10,
        "unknown": 11,
    }
    return jobs[job]


def marital_status(status: str) -> int:
    statuses = {
        "divorced": 0,
        "married": 1,
        "single": 2,
        "unknown": 3,
    }
    return statuses[status]


def education_level(level: str) -> int:
    ed_level = {
        "basic.4y": 0,
        "basic.6y": 1,
        "basic.9y": 2,
        "high.school": 3,
        "illiterate": 4,
        "professional.course": 5,
        "university.degree": 6,
        "unknown": 7,
    }

    return ed_level[level]


def default_status(status: str) -> int:
    statuses = {
        "no": 0,
        "yes": 1,
        "unknown": 2,
    }
    return statuses[status]


def housing_loan(status: str) -> int:
    statuses = {
        "no": 0,
        "yes": 1,
        "unknown": 2,
    }
    return statuses[status]


def loan_status(status: str) -> int:
    statuses = {
        "no": 0,
        "yes": 1,
        "unknown": 2,
    }
    return statuses[status]


def month_index(month: str) -> int:
    months = {
        "jan": 0,
        "feb": 1,
        "mar": 2,
        "apr": 3,
        "may": 4,
        "jun": 5,
        "jul": 6,
        "aug": 7,
        "sep": 8,
        "oct": 9,
        "nov": 10,
        "dec": 11,
    }
    return months[month]

def day_of_week_index(day: str) -> int:
    days = {
        "mon": 0,
        "tue": 1,
        "wed": 2,
        "thu": 3,
        "fri": 4,
    }
    return days[day]

def poutcome_status(status: str) -> int:
    statuses = {
        "failure": 0,
        "nonexistent": 1,
        "success": 2,
    }
    return statuses[status]

def contact_type(contact: str) -> int:
    contacts = {
        "cellular": 0,
        "telephone": 1,
    }
    return contacts[contact]

def result_status(status: str) -> int:
    statuses = {
        "no": 0,
        "yes": 1,
    }
    return statuses[status]


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
            "y": lambda x: result_status(x),
            "job": lambda x: index_job(x) if x != na_values else np.nan,
            "marital": lambda x: marital_status(x) if x != na_values else np.nan,
            "education": lambda x: education_level(x) if x != na_values else np.nan,
            "default": lambda x: default_status(x) if x != na_values else np.nan,
            "housing": lambda x: housing_loan(x) if x != na_values else np.nan,
            "loan": lambda x: loan_status(x) if x != na_values else np.nan,
            "month": lambda x: month_index(x),
            "day_of_week": lambda x: day_of_week_index(x),
            "contact": lambda x: contact_type(x),
            "poutcome": lambda x: poutcome_status(x),
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
    pass


def correlation_matrix(data: pd.DataFrame):
    corr_matrix = data.corr("pearson")
    print(f"Correlation matrix:\n{corr_matrix}")
    return corr_matrix

def plot_correlation_matrix(corr_matrix: pd.DataFrame):
    """
    Plot correlation matrix with heatmap
    """
    plt.figure(figsize=(CORRELATION_MATRIX_SIZE, CORRELATION_MATRIX_SIZE))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
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
    data_path = __file__.replace("main.py", "data/bank-additional/bank-additional/bank-additional-full.csv")
    data = load_data(
        data_path,
        usecols=cols,
    )
    data_summary(data)

    mrx = correlation_matrix(data)

    plot_correlation_matrix(mrx)

if __name__ == "__main__":
    main()
