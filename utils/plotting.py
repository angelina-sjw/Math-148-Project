import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: list) -> None:
    n_numeric = len(numeric_cols)
    ncols = 3
    nrows = int(np.ceil(n_numeric / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=50, color='skyblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    for ax in axes[n_numeric:]:
        fig.delaxes(ax)
    plt.tight_layout()
    plt.show()

def plot_categorical_bars(df: pd.DataFrame, categorical_cols: list) -> None:
    n_categorical = len(categorical_cols)
    ncols = 3
    nrows = int(np.ceil(n_categorical / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()
    for i, col in enumerate(categorical_cols):
        value_counts = df[col].value_counts().sort_index()
        axes[i].bar(value_counts.index.astype(str), value_counts.values, color='salmon', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
    for ax in axes[n_categorical:]:
        fig.delaxes(ax)
    plt.tight_layout()
    plt.show()

def plot_target_distribution(y: pd.Series, title: str = "Target Variable Distribution") -> None:
    target_counts = y.value_counts().sort_index()
    plt.figure(figsize=(4, 3))
    plt.bar(target_counts.index.astype(str), target_counts.values, color='lightgreen', edgecolor='black')
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_labels: list) -> None:
    from sklearn.metrics import confusion_matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=class_labels)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=False, cmap='Blues',
                     xticklabels=class_labels, yticklabels=class_labels)
    threshold = (conf_matrix.min() + conf_matrix.max()) / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = conf_matrix[i, j]
            text_color = "white" if value > threshold else "black"
            ax.text(j + 0.5, i + 0.5, value, ha='center', va='center', color=text_color, fontsize=12)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()
