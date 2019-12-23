# Project Structure

## Dataset

The dataset can be found at: https://www.kaggle.com/asaumya/healthcare-dataset-stroke-data

## Environment

The environment of this project is prepared through Conda. To reproduce the environment, just call:

```
conda create -f environment.yml
conda activate stroke
```

## Tasks

The dataset preparation and training of the models are organized through `luigi`. It's a framework used to organize a graph of tasks.

We have two important tasks: PreProcessAndSplitDataset and a subclass of BaseModelTraining. The first one preprocess and split the dataset according to the given parameters. The second one trains a model and create a folder with the metrics, plots and so on. The BaseModelTraining also has the parameters used by PreProcessAndSplitDataset. It's used to propagate these parameters and easily change the way the dataset is preprocessed and split.

# Reproducing the best model

To reproduce the best model, it's only necessary to call:

```
PYTHONPATH="." luigi --module stroke.model RandomForestClassifierTraining --class-weight balanced --sampling-strategy none --max-depth 10 --n-estimators 200 --min-samples-split 500 --min-samples-leaf 200 --oob-score --min-impurity-decrease 1e-3 --local-scheduler
```
