import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    df = pd.read_csv(path)
    return df


def preprocess_data(df):

    # Цільова змінна
    df = df.dropna(subset=["RainTomorrow"])

    df["RainTomorrow"] = df["RainTomorrow"].map({"Yes":1, "No":0})
    df["RainToday"] = df["RainToday"].map({"Yes":1, "No":0})

    # Заповнення пропусків
    numeric_cols = df.select_dtypes(include=["float64","int64"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        df[col] = df[col].fillna("Unknown")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    X = df.drop("RainTomorrow", axis=1)
    y = df["RainTomorrow"]

    return X, y


def plot_confusion_matrix(y_test, y_pred):

    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("confusion_matrix.png")
    plt.close()


def plot_feature_importance(model, X):

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10,6))
    plt.title("Feature Importance")
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), X.columns[indices], rotation=90)

    plt.tight_layout()
    plt.savefig("feature_importance.png")
    plt.close()


def train(args):

    df = load_data(args.data_path)

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=42
    )

    mlflow.set_experiment("Weather_Rain_Prediction")

    with mlflow.start_run():

        mlflow.set_tag("author", "student")
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("dataset", "WeatherAUS")

        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Прогноз та метрики для тестової вибірки
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)

        # Прогноз та метрики для тренувальної вибірки
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)

        # Логування параметрів
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_param("test_size", args.test_size)

        # Логування метрик
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("train_accuracy", train_accuracy)
        mlflow.log_metric("train_f1_score", train_f1)

        # графіки
        plot_confusion_matrix(y_test, y_pred)
        plot_feature_importance(model, X)

        mlflow.log_artifact("confusion_matrix.png")
        mlflow.log_artifact("feature_importance.png")

        # логування моделі
        mlflow.sklearn.log_model(model, "random_forest_model")

        print("Train Accuracy:", train_accuracy)
        print("Train F1 Score:", train_f1)
        print("Test Accuracy:", test_accuracy)
        print("Test F1 Score:", test_f1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_path",
        type=str,
        default="data/raw/WeatherAUS.csv"
    )

    parser.add_argument(
        "--n_estimators",
        type=int,
        default=100
    )

    parser.add_argument(
        "--max_depth",
        type=int,
        default=10
    )

    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2
    )

    args = parser.parse_args()

    train(args)