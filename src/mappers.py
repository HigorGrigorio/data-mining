# -----------------------------------------------------------------------------
# (C) 2024 Andre Conde (andre.conde100@gmail.com)  (MIT License)
# -----------------------------------------------------------------------------


from functools import lru_cache


class BaseMapper:
    """
    Base class for mappers.
    """

    __column__ = None
    __data__ = None

    def __init__(self):
        pass

    @classmethod
    def map(cls, value):
        if cls.__data__ is None:
            raise NotImplementedError("The __data__ attribute must be implemented.")

        return cls.__data__[value]

    @classmethod
    def revert(cls, value):
        if cls.__data__ is None:
            raise NotImplementedError("The __data__ attribute must be implemented.")

        return list(cls.__data__.keys())[list(cls.__data__.values()).index(value)]

    @classmethod
    def map_list(cls, values):
        return [cls.map(value) for value in values]

    @classmethod
    def revert_list(cls, values):
        return [cls.revert(value) for value in values]

    @staticmethod
    @lru_cache
    def get_mapper(name: str):
        classes = BaseMapper.__subclasses__()

        for cls in classes:
            if cls.__column__ == name:
                return cls

        raise ValueError(f"Mapper {name} not found.")


class JobMapper(BaseMapper):
    """
    A class that provides mapping functions for the job attribute.
    """

    __column__ = "job"
    __data__ = {
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
    }


class MaritalStatusMapper(BaseMapper):
    """
    A class that provides mapping functions for the marital status attribute.
    """

    __column__ = "marital"
    __data__ = {
        "divorced": 0,
        "married": 1,
        "single": 2,
    }


class EducationLevelMapper(BaseMapper):
    """
    A class that provides mapping functions for the education level attribute.
    """

    __column__ = "education"
    __data__ = {
        "basic.4y": 0,
        "basic.6y": 1,
        "basic.9y": 2,
        "high.school": 3,
        "illiterate": 4,
        "professional.course": 5,
        "university.degree": 6,
    }


class DefaultStatusMapper(BaseMapper):
    """
    A class that provides mapping functions for the default status attribute.
    """

    __column__ = "default"
    __data__ = {
        "no": 0,
        "yes": 1,
    }


class HousingLoanMapper(BaseMapper):
    """
    A class that provides mapping functions for the housing loan attribute.
    """

    __column__ = "housing"
    __data__ = {
        "no": 0,
        "yes": 1,
    }


class LoanStatusMapper(BaseMapper):
    """
    A class that provides mapping functions for the loan status attribute.
    """

    __column__ = "loan"
    __data__ = {
        "no": 0,
        "yes": 1,
    }


class MonthMapper(BaseMapper):
    """
    A class that provides mapping functions for the month attribute.
    """

    __column__ = "month"
    __data__ = {
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


class DayOfWeekMapper(BaseMapper):
    """
    A class that provides mapping functions for the day of the week attribute.
    """

    __column__ = "day_of_week"
    __data__ = {
        "mon": 0,
        "tue": 1,
        "wed": 2,
        "thu": 3,
        "fri": 4,
    }


class PoutcomeStatusMapper(BaseMapper):
    """
    A class that provides mapping functions for the poutcome status attribute.
    """

    __column__ = "poutcome"
    __data__ = {
        "failure": 0,
        "nonexistent": 1,
        "success": 2,
    }


class ContactTypeMapper(BaseMapper):
    """
    A class that provides mapping functions for the contact type attribute.
    """

    __column__ = "contact"
    __data__ = {
        "cellular": 0,
        "telephone": 1,
    }


class ResultStatusMapper(BaseMapper):
    """
    A class that provides mapping functions for the result status attribute.
    """

    __column__ = "result_status"
    __data__ = {
        "no": 0,
        "yes": 1,
    }
