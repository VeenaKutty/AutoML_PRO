def detect_problem_type(y):
    if y.dtype == "object" or y.nunique() < 20:
        return "classification"
    else:
        return "regression"