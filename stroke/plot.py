from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

sns.set()


def plot_confusion_matrix(targets: np.ndarray, preds: np.ndarray, class_names, figsize=(10, 7),
                          fontsize=12) -> Figure:
    confusion_matrix_ = confusion_matrix(targets, preds)

    df_cm = pd.DataFrame(
        confusion_matrix_, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    heatmap = sns.heatmap(df_cm, annot=True, ax=ax, fmt="d", cmap="Blues")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='center', fontsize=fontsize)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig


def plot_roc_curve(targets: np.ndarray, probas: np.ndarray, figsize=(10, 7)) -> Figure:
    fpr, tpr, _ = roc_curve(targets, probas)
    roc_auc = roc_auc_score(targets, probas)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    ax.set_title('Receiver Operating Characteristic')
    ax.plot(fpr, tpr, 'b', label='AUC = %0.6f' % roc_auc)
    ax.legend(loc='lower right')
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')

    return fig


def plot_decision_tree(tree: DecisionTreeClassifier, feature_names: List[str], class_names: List[str],
                       figsize=(30, 22)) -> Figure:
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plot_tree(tree, feature_names=feature_names, class_names=class_names, ax=ax)

    return fig
