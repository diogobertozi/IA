import matplotlib.pyplot as plt
import seaborn as sns

def scatter_plot(data, x_col, y_col, clusters=None, title='Scatter Plot', xlabel='X-axis', ylabel='Y-axis'):
    """
    Create a scatter plot to visualize clusters in the data.

    Parameters:
    - data: DataFrame containing the data to plot.
    - x_col: Column name for the x-axis.
    - y_col: Column name for the y-axis.
    - clusters: Optional; array-like of cluster labels for coloring the points.
    - title: Title of the plot.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    """
    plt.figure(figsize=(10, 6))
    
    if clusters is not None:
        sns.scatterplot(data=data, x=x_col, y=y_col, hue=clusters, palette='viridis', alpha=0.7)
    else:
        sns.scatterplot(data=data, x=x_col, y=y_col, color='blue', alpha=0.7)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(title='Clusters' if clusters is not None else None)
    plt.grid(True)
    plt.show()