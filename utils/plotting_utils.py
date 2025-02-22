import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Any

def plot_numeric_histograms(df: pd.DataFrame, numeric_cols: List[str]) -> None:
    """
    Plot histograms for numeric columns in the DataFrame.
    
    Each numeric column is displayed in a separate subplot arranged in a grid.
    """
    n_numeric = len(numeric_cols)
    ncols = 3  # Number of columns per row in the subplot grid
    nrows = int(np.ceil(n_numeric / ncols))
    
    # Create subplots with the calculated grid size
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()  # Flatten to simplify iteration
    
    # Plot histogram for each numeric column
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=50, color='skyblue', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")
    
    # Remove unused subplots if any
    for ax in axes[n_numeric:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()


def plot_categorical_bars(df: pd.DataFrame, categorical_cols: List[str]) -> None:
    """
    Plot bar charts for categorical columns in the DataFrame.
    
    Each categorical column is displayed in a separate subplot arranged in a grid.
    """
    n_categorical = len(categorical_cols)
    ncols = 3  # Number of columns per row in the subplot grid
    nrows = int(np.ceil(n_categorical / ncols))
    
    # Create subplots with the calculated grid size
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()  # Flatten to simplify iteration
    
    # Plot bar chart for each categorical column
    for i, col in enumerate(categorical_cols):
        # Sort value counts by index for consistent ordering
        value_counts = df[col].value_counts().sort_index()
        axes[i].bar(value_counts.index.astype(str), value_counts.values,
                    color='salmon', edgecolor='black')
        axes[i].set_title(col)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Count")
    
    # Remove unused subplots if any
    for ax in axes[n_categorical:]:
        fig.delaxes(ax)
    
    plt.tight_layout()
    plt.show()


def plot_target_distribution(y: pd.Series, title: str = "Target Variable Distribution") -> None:
    """
    Plot the distribution of a target variable as a bar chart.
    """
    target_counts = y.value_counts().sort_index()
    plt.figure(figsize=(4, 3))
    plt.bar(target_counts.index.astype(str), target_counts.values,
            color='lightgreen', edgecolor='black')
    plt.title(title)
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_test: Any, y_pred: Any, class_labels: List[str]) -> None:
    """
    Plot a confusion matrix using a heatmap.
    
    The confusion matrix is computed from true and predicted labels and then 
    visualized using seaborn's heatmap. Each cell is annotated with the count.
    """
    from sklearn.metrics import confusion_matrix
    # Compute the confusion matrix using provided labels
    conf_matrix = confusion_matrix(y_test, y_pred, labels=class_labels)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(conf_matrix, annot=False, cmap='Blues',
                     xticklabels=class_labels, yticklabels=class_labels)
    
    # Define a threshold to adjust text color for better visibility
    threshold = (conf_matrix.min() + conf_matrix.max()) / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            value = conf_matrix[i, j]
            text_color = "white" if value > threshold else "black"
            ax.text(j + 0.5, i + 0.5, value, ha='center', va='center',
                    color=text_color, fontsize=12)
    
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()