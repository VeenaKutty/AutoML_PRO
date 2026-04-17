from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def get_model_and_params(config, problem_type):
    if problem_type == "classification":
        return {
            LogisticRegression(): config["models"]["classification"]["LogisticRegression"],
            RandomForestClassifier(): config["models"]["classification"]["RandomForestClassifier"]
        }
    else:
        return {
            LinearRegression(): {},
            RandomForestRegressor(): config["models"]["regression"]["RandomForestRegressor"]
        }

from sklearn.model_selection import GridSearchCV

def find_best_model(models, X, y, cv):
    best_model = None
    best_score = -999
    leaderboard = []

    for model, params in models.items():
        grid = GridSearchCV(
                            model,
                            params if params else {},
                            cv=cv,
                            n_jobs=-1,
                            error_score="raise"  
)
        grid.fit(X, y)

        score = grid.best_score_

        leaderboard.append({
            "model": type(model).__name__,
            "score": score
        })

        if score > best_score:
            best_score = score
            best_model = grid.best_estimator_
    return best_model, best_score, leaderboard