import abc
import json
import os
import pickle
from collections import OrderedDict
from typing import List

import luigi
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier

from stroke.data_preparation import PreProcessAndSplitDataset, BASE_DIR
from stroke.plot import plot_roc_curve, plot_confusion_matrix, plot_decision_tree


class BaseModelTraining(luigi.Task, metaclass=abc.ABCMeta):
    test_size: float = luigi.FloatParameter(default=0.2)
    dataset_split_method: str = luigi.ChoiceParameter(choices=["holdout", "k_fold"], default="holdout")
    n_splits: int = luigi.IntParameter(default=5)
    split_index: int = luigi.IntParameter(default=0)
    sampling_strategy: str = luigi.ChoiceParameter(choices=["oversample", "undersample", "none"], default="none")

    smoking_status_imputation_strategy: str = luigi.ChoiceParameter(choices=["mode", "mode_by_gender"],
                                                                    default="mode")
    bmi_imputation_strategy: str = luigi.ChoiceParameter(choices=["mean", "mean_by_gender"], default="mean")
    normalize_numerical_features: bool = luigi.BoolParameter(default=False)

    excluded_features: List[str] = luigi.ListParameter(default=[])

    class_weight: str = luigi.ChoiceParameter(choices=["balanced", "none"], default="none")

    decision_threshold: float = luigi.FloatParameter(default=0.5)

    seed: int = luigi.IntParameter(default=42)

    def requires(self):
        return PreProcessAndSplitDataset(test_size=self.test_size, sampling_strategy=self.sampling_strategy,
                                         smoking_status_imputation_strategy=self.smoking_status_imputation_strategy,
                                         bmi_imputation_strategy=self.bmi_imputation_strategy,
                                         normalize_numerical_features=self.normalize_numerical_features,
                                         seed=self.seed)

    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "experiments", self.__class__.__name__, self.task_id))

    def _get_params(self) -> dict:
        return self.param_kwargs

    def _save_params(self):
        params = self._get_params()
        with open(os.path.join(self.output().path, "params.json"), "w") as params_file:
            json.dump(params, params_file, default=lambda o: dict(o), indent=4)

    def _save_model(self, model: BaseEstimator):
        with open(os.path.join(self.output().path, "model.pkl"), "wb") as pickle_file:
            pickle.dump(model, pickle_file)

    def _save_metrics(self, metrics: dict):
        with open(os.path.join(self.output().path, "metrics.json"), "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)

    @property
    def train_df(self) -> pd.DataFrame:
        if not hasattr(self, "_train_df"):
            self._train_df = pd.read_csv(self.input()[0].path).drop(columns=list(self.excluded_features))
        return self._train_df

    @property
    def test_df(self) -> pd.DataFrame:
        if not hasattr(self, "_test_df"):
            self._test_df = pd.read_csv(self.input()[1].path).drop(columns=list(self.excluded_features))
        return self._test_df

    @property
    def feature_names(self) -> List[str]:
        if not hasattr(self, "_feature_names"):
            self._feature_names = list(self.train_df.drop(columns=["id", "stroke"]).columns)
        return self._feature_names

    @abc.abstractmethod
    def create_model(self) -> BaseEstimator:
        pass

    def run(self):
        os.makedirs(self.output().path)

        np.random.seed(self.seed)

        self._save_params()

        X_train = self.train_df.drop(columns=["id", "stroke"]).values
        y_train = self.train_df["stroke"].values
        X_test = self.test_df.drop(columns=["id", "stroke"]).values
        y_test = self.test_df["stroke"].values

        self.model = self.create_model()

        self.model.fit(X_train, y_train)
        self._save_model(self.model)

        y_proba_train: np.ndarray = self.model.predict_proba(X_train)[:, 1]
        y_pred_train: np.ndarray = (y_proba_train > self.decision_threshold).astype(np.int)
        y_proba_test: np.ndarray = self.model.predict_proba(X_test)[:, 1]
        y_pred_test: np.ndarray = (y_proba_test > self.decision_threshold).astype(np.int)

        metrics = OrderedDict([
            ("train_acc", accuracy_score(y_train, y_pred_train)),
            ("test_acc", accuracy_score(y_test, y_pred_test)),
            ("train_precision", precision_score(y_train, y_pred_train)),
            ("test_precision", precision_score(y_test, y_pred_test)),
            ("train_recall", recall_score(y_train, y_pred_train)),
            ("test_recall", recall_score(y_test, y_pred_test)),
            ("train_f1_score", f1_score(y_train, y_pred_train)),
            ("test_f1_score", f1_score(y_test, y_pred_test)),
            ("train_roc_auc", roc_auc_score(y_train, y_proba_train)),
            ("test_roc_auc", roc_auc_score(y_test, y_proba_test)),
        ])

        self._save_metrics(metrics)

        class_names = ["No Stroke", "Stroke"]
        plot_confusion_matrix(y_train, y_pred_train, class_names).savefig(
            os.path.join(self.output().path, "train_confusion_matrix.png"))
        plot_confusion_matrix(y_test, y_pred_test, class_names).savefig(
            os.path.join(self.output().path, "test_confusion_matrix.png"))

        plot_roc_curve(y_train, y_proba_train).savefig(os.path.join(self.output().path, "train_roc_curve.png"))
        plot_roc_curve(y_test, y_proba_test).savefig(os.path.join(self.output().path, "test_roc_curve.png"))


class LogisticRegressionTraining(BaseModelTraining):
    normalize_numerical_features: bool = luigi.BoolParameter(default=False)

    penalty: str = luigi.ChoiceParameter(choices=["l1", "l2", "elasticnet", "none"], default="l2")
    dual: bool = luigi.BoolParameter(default=False)
    tol: float = luigi.FloatParameter(default=1e-4)
    c: float = luigi.FloatParameter(default=1.0)
    solver: str = luigi.ChoiceParameter(choices=["newton-cg", "lbfgs", "liblinear", "sag", "saga"], default="lbfgs")
    max_iter: int = luigi.IntParameter(default=100)

    def create_model(self) -> LogisticRegression:
        return LogisticRegression(penalty=self.penalty, dual=self.dual, tol=self.tol, C=self.c,
                                  solver=self.solver, max_iter=self.max_iter,
                                  class_weight=self.class_weight if self.class_weight != "none" else None, verbose=1,
                                  random_state=self.seed)

    def _save_model_parameters(self, metrics: dict):
        with open(os.path.join(self.output().path, "model_parameters.json"), "w") as model_parameters_file:
            json.dump(metrics, model_parameters_file, indent=4)

    def run(self):
        super().run()

        self._save_model_parameters({
            "feature_names": self.feature_names,
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
        })


class DecisionTreeClassifierTraining(BaseModelTraining):
    criterion: str = luigi.ChoiceParameter(choices=["gini", "entropy"], default="gini")
    splitter: str = luigi.ChoiceParameter(choices=["best", "random"], default="best")
    max_depth: int = luigi.IntParameter(default=None)
    min_samples_split: int = luigi.IntParameter(default=2)
    min_samples_leaf: int = luigi.IntParameter(default=1)
    min_weight_fraction_leaf: float = luigi.FloatParameter(default=0.0)
    max_features: int = luigi.IntParameter(default=None)
    max_leaf_nodes: int = luigi.IntParameter(default=None)
    min_impurity_decrease: float = luigi.FloatParameter(default=0.0)
    ccp_alpha: float = luigi.FloatParameter(default=0.0)

    def create_model(self) -> BaseEstimator:
        return DecisionTreeClassifier(criterion=self.criterion, splitter=self.splitter, max_depth=self.max_depth,
                                      min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf,
                                      min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                      max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes,
                                      min_impurity_decrease=self.min_impurity_decrease, ccp_alpha=self.ccp_alpha,
                                      class_weight=self.class_weight if self.class_weight != "none" else None,
                                      random_state=self.seed)

    def _save_feature_importances(self, metrics: dict):
        with open(os.path.join(self.output().path, "feature_importances.json"), "w") as file:
            json.dump(metrics, file, indent=4)

    def run(self):
        super().run()

        self._save_feature_importances({
            "feature_names": self.feature_names,
            "feature_importances": self.model.feature_importances_.tolist(),
        })

        if isinstance(self.model, DecisionTreeClassifier):
            plot_decision_tree(self.model, self.feature_names, ["No Stroke", "Stroke"]).savefig(
                os.path.join(self.output().path, "tree.png"))


class RandomForestClassifierTraining(DecisionTreeClassifierTraining):
    n_estimators: int = luigi.IntParameter(default=100)
    oob_score: bool = luigi.BoolParameter(default=False)

    def create_model(self) -> BaseEstimator:
        return RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                      max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf,
                                      min_weight_fraction_leaf=self.min_weight_fraction_leaf,
                                      max_features=self.max_features, max_leaf_nodes=self.max_leaf_nodes,
                                      min_impurity_decrease=self.min_impurity_decrease, ccp_alpha=self.ccp_alpha,
                                      class_weight=self.class_weight if self.class_weight != "none" else None,
                                      oob_score=self.oob_score, verbose=1, random_state=self.seed)
