from sklearn.feature_selection import SelectKBest, f_classif, f_regression

def get_feature_selector(problem_type, k):
    if problem_type == "classification":
        return SelectKBest(score_func=f_classif, k=k)
    return SelectKBest(score_func=f_regression, k=k)