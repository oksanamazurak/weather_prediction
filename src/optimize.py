import os
import hydra
import mlflow
import optuna
import pandas as pd
import json

from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import mlflow.sklearn


def objective(trial, cfg, X, y):

    with mlflow.start_run(nested=True):

        # ---- Tags ----
        mlflow.set_tags({
            "trial_number": trial.number,
            "sampler": cfg.hpo.sampler,
            "model_type": cfg.model.model_type,
            "seed": cfg.seed
        })

        # ---- Model selection ----
        if cfg.model.model_type == "random_forest":

            model = RandomForestClassifier(
                n_estimators=trial.suggest_int("n_estimators", 50, 300),
                max_depth=trial.suggest_int("max_depth", 3, 20),
                min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
                random_state=cfg.seed,
                n_jobs=-1
            )

        elif cfg.model.model_type == "logistic_regression":

            model = LogisticRegression(
                C=trial.suggest_float("C", 0.01, 10.0, log=True),
                max_iter=1000,
                random_state=cfg.seed
            )

        else:
            raise ValueError("Unsupported model type")

        # ---- Train/Test split ----
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=cfg.seed,
            stratify=y
        )

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        accuracy = accuracy_score(y_test, preds)

        mlflow.log_params(trial.params)
        mlflow.log_metric("accuracy", accuracy)

        return accuracy


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig):

    os.chdir(hydra.utils.get_original_cwd())


    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    df = pd.read_csv(cfg.data.train_path)

    X = df.drop("RainTomorrow", axis=1)
    y = df["RainTomorrow"]

    with mlflow.start_run():

        config_dict = OmegaConf.to_container(cfg, resolve=True)
        mlflow.log_dict(config_dict, "config_used.json")

        if cfg.hpo.sampler == "tpe":
            sampler = optuna.samplers.TPESampler(seed=cfg.seed)

        elif cfg.hpo.sampler == "random":
            sampler = optuna.samplers.RandomSampler(seed=cfg.seed)

        elif cfg.hpo.sampler == "grid":

            search_space = {
                "n_estimators": [100, 200],
                "max_depth": [5, 10],
                "min_samples_split": [2, 5]
            }

            sampler = optuna.samplers.GridSampler(search_space)

        else:
            raise ValueError("Unsupported sampler")

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler
        )

        study.optimize(
            lambda trial: objective(trial, cfg, X, y),
            n_trials=cfg.hpo.n_trials
        )

        mlflow.log_metric("best_accuracy", study.best_value)
        mlflow.log_params(study.best_params)

        mlflow.log_dict(study.best_params, "best_params.json")

        if cfg.model.model_type == "random_forest":
            final_model = RandomForestClassifier(
                **study.best_params,
                random_state=cfg.seed,
                n_jobs=-1
            )
        else:
            final_model = LogisticRegression(
                **study.best_params,
                max_iter=1000
            )

        final_model.fit(X, y)

        mlflow.sklearn.log_model(final_model, "final_model")

        print("=================================")
        print("Best parameters:", study.best_params)
        print("Best accuracy:", study.best_value)
        print("=================================")


if __name__ == "__main__":
    main()