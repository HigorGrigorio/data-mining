# -----------------------------------------------------------------------------
# (C) 2024 Andre Conde (andre.conde100@gmail.com)  (MIT License)
# -----------------------------------------------------------------------------


class Mapper:
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
