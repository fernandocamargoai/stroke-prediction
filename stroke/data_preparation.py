import itertools
import os
from typing import Tuple

import luigi
import numpy as np
import pandas as pd
from dython.nominal import numerical_encoding
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler

BASE_DIR: str = os.path.join("output")
DATASET_DIR: str = os.path.join(BASE_DIR, "dataset")


class CheckDataset(luigi.Task):
    def output(self):
        return luigi.LocalTarget(os.path.join(BASE_DIR, "train_2v.csv"))

    def run(self):
        raise AssertionError(
            f"The dataset is expected to be at {self.output().path}")


class PreProcessAndSplitDataset(luigi.Task):
    test_size: float = luigi.FloatParameter(default=0.2)
    dataset_split_method: str = luigi.ChoiceParameter(choices=["holdout", "k_fold"], default="holdout")
    n_splits: int = luigi.IntParameter(default=5)
    split_index: int = luigi.IntParameter(default=0)
    sampling_strategy: str = luigi.ChoiceParameter(choices=["oversample", "undersample", "none"], default="none")

    smoking_status_imputation_strategy: str = luigi.ChoiceParameter(choices=["mode", "mode_by_gender"],
                                                                    default="mode")
    bmi_imputation_strategy: str = luigi.ChoiceParameter(choices=["mean", "mean_by_gender"], default="mean")
    normalize_numerical_features: bool = luigi.BoolParameter(default=False)

    seed: int = luigi.IntParameter(default=42)

    def requires(self):
        return CheckDataset()

    def output(self):
        task_hash = self.task_id.split("_")[-1]
        return luigi.LocalTarget(os.path.join(DATASET_DIR, "train_%s.csv" % task_hash)), \
               luigi.LocalTarget(os.path.join(DATASET_DIR, "test_%s.csv" % task_hash))

    def _kfold_split(self, df) -> Tuple[pd.DataFrame, pd.DataFrame]:
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        train_indices, val_indices = next(
            itertools.islice(skf.split(df, df["stroke"]),
                             self.split_index, self.split_index + 1))
        train_df, test_df = df.iloc[train_indices], df.iloc[val_indices]
        return train_df, test_df

    def _balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        sampler = dict(oversample=RandomOverSampler,
                       undersample=RandomUnderSampler)

        random_sampler_cls = sampler.get(self.sampling_strategy)

        if random_sampler_cls is None:
            return df

        index_resampled = df.index
        random_sampler = random_sampler_cls(random_state=self.seed)
        index_resampled, _ = random_sampler.fit_sample(np.array(index_resampled).reshape(-1, 1),
                                                       df.loc[index_resampled]["stroke"])
        index_resampled = index_resampled.flatten()

        return df.loc[index_resampled]

    def run(self):
        os.makedirs(DATASET_DIR, exist_ok=True)

        df = pd.read_csv(self.input().path)

        if self.dataset_split_method == "holdout":
            train_df, test_df = train_test_split(df, test_size=self.test_size, stratify=df["stroke"],
                                                 random_state=self.seed)
        else:
            train_df, test_df = self._kfold_split(df)

        if self.smoking_status_imputation_strategy == "mode":
            smoking_status_mode = train_df["smoking_status"].mode()
            train_df["smoking_status"] = train_df["smoking_status"].fillna(smoking_status_mode)
            test_df["smoking_status"] = test_df["smoking_status"].fillna(smoking_status_mode)
        elif self.smoking_status_imputation_strategy == "mode_by_gender":
            male_smoking_status_mode = train_df.loc[train_df["gender"] == "Male", "smoking_status"].mode()
            female_smoking_status_mode = train_df.loc[train_df["gender"] == "Female", "smoking_status"].mode()
            other_smoking_status_mode = train_df.loc[train_df["gender"] == "Other", "smoking_status"].mode()

            smoking_status_mode_dict = dict(Male=male_smoking_status_mode, Female=female_smoking_status_mode,
                                            Other=other_smoking_status_mode)

            train_df["smoking_status"] = train_df["smoking_status"].apply(
                lambda ss: smoking_status_mode_dict[ss] if pd.isna(ss) else ss)
            test_df["smoking_status"] = test_df["smoking_status"].apply(
                lambda ss: smoking_status_mode_dict[ss] if pd.isna(ss) else ss)

        if self.bmi_imputation_strategy == "mean":
            bmi_mean = train_df["bmi"].mean()
            train_df["bmi"] = train_df["bmi"].fillna(bmi_mean)
            test_df["bmi"] = test_df["bmi"].fillna(bmi_mean)
        elif self.bmi_imputation_strategy == "mean_by_gender":
            male_bmi_mean = train_df.loc[train_df["gender"] == "Male", "bmi"].mean()
            female_bmi_mean = train_df.loc[train_df["gender"] == "Female", "bmi"].mean()
            other_bmi_mean = train_df.loc[train_df["gender"] == "Other", "bmi"].mean()

            bmi_mean_dict = dict(Male=male_bmi_mean, Female=female_bmi_mean,
                                 Other=other_bmi_mean)

            train_df["bmi"] = train_df["bmi"].apply(
                lambda bmi: bmi_mean_dict[bmi] if pd.isna(bmi) else bmi)
            test_df["bmi"] = test_df["bmi"].apply(
                lambda bmi: bmi_mean_dict[bmi] if pd.isna(bmi) else bmi)

        nominal_columns = ["gender", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type",
                           "smoking_status"]
        train_df = numerical_encoding(train_df, nominal_columns=nominal_columns, nan_strategy="SKIP")
        test_df = numerical_encoding(test_df, nominal_columns=nominal_columns, nan_strategy="SKIP")

        if self.normalize_numerical_features:
            age_scaler = StandardScaler()
            age_scaler.fit(train_df["age"])
            avg_glucose_level_scaler = StandardScaler()
            avg_glucose_level_scaler.fit(train_df["avg_glucose_level"])
            bmi_scaler = StandardScaler()
            bmi_scaler.fit(train_df["bmi"])

            train_df["age"] = age_scaler.transform(train_df["age"])
            test_df["age"] = age_scaler.transform(test_df["age"])
            train_df["avg_glucose_level"] = avg_glucose_level_scaler.transform(train_df["avg_glucose_level"])
            test_df["avg_glucose_level"] = avg_glucose_level_scaler.transform(test_df["avg_glucose_level"])
            train_df["bmi"] = bmi_scaler.transform(train_df["bmi"])
            test_df["bmi"] = bmi_scaler.transform(test_df["bmi"])

        if self.sampling_strategy != "none":
            train_df = self._balance_dataset(train_df)

        train_df.to_csv(self.output()[0].path, index=False)
        test_df.to_csv(self.output()[1].path, index=False)
