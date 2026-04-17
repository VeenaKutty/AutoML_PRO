import pandas as pd
import joblib
import json

from sklearn.model_selection import StratifiedKFold
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.config_loader import load_config
from src.problem_type import detect_problem_type
from src.preprocessing import get_preprocessor
from src.feature_selection import get_feature_selector
from src.model_selection import get_model_and_params, find_best_model
from src.outlier import remove_outliers


def train(df, target):
    config = load_config()


    df = df.dropna(subset=[target])


    df = remove_outliers(df)


    X = df.drop(columns=[target])
    y = df[target]

    unique_classes = y.nunique()

    if unique_classes < 2:
        raise ValueError(f"Target column has only {unique_classes} class. Need at least 2 classes for classification.")

    problem_type = detect_problem_type(y)

    preprocessor = get_preprocessor(X)


    k = min(config["feature_selection"]["k"], X.shape[1])
    selector = get_feature_selector(problem_type, k)


    models = get_model_and_params(config, problem_type)
    

    if problem_type == "classification":
        cv = StratifiedKFold(n_splits=min(5, y.value_counts().min()))
    else:
        cv = min(config["cv_folds"], len(y))

    X_temp = preprocessor.fit_transform(X)

    best_model, score, leaderboard = find_best_model(models, X_temp, y, cv)

    steps = [
        ("preprocessor", preprocessor),
    ]

    if problem_type == "classification" and config["smote"]:
        if len(set(y)) > 1 and len(y) > 10:
            steps.append(("smote", SMOTE(k_neighbors=3)))

    steps.append(("selector", selector))
    steps.append(("model", best_model))

    pipeline = Pipeline(steps)

    pipeline.fit(X, y)

    joblib.dump(pipeline, "artifacts/model.pkl")

    with open("artifacts/metrics.json", "w") as f:
        json.dump(
            {
                "score": float(score),
                "problem_type": problem_type
            },
            f
        )

    return score, problem_type, leaderboard