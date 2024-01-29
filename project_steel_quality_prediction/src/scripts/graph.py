import numpy as np
import matplotlib.pyplot as plt
import os


def box_plot(df, data_set, path_for_figures, figure_name):
    """
    Creates and saves box plots for each column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - data_set (str): Identifier for the dataset.
    - path_for_figures (str): Path to save the figures.
    - figure_name (str): Name of the saved figure.
    """
    plt.figure(figsize = (12, 16))

    for i, column in enumerate(df.columns[1:]):
        plt.subplot(7, 3, i + 1)
        plt.grid(True)
        plt.boxplot(df[column])
        plt.title(f'Box Plot ({data_set}) - {column}')

    plt.subplots_adjust(hspace = 0.7, wspace = 0.5)

    box_plots_path = os.path.join(path_for_figures, figure_name)
    plt.savefig(box_plots_path)


def scatter_plot(df, data_set, path_for_figures, figure_name, withOutput = False):
    """
    Creates and saves scatter plots for each column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - data_set (str): Identifier for the dataset.
    - path_for_figures (str): Path to save the figures.
    - figure_name (str): Name of the saved figure.
    - withOutput (bool): Flag to include output in scatter plots.
    """
    plt.figure(figsize = (12, 16))

    for i, column in enumerate(df.columns[1:]):
        plt.subplot(7, 3, i + 1)
        if (withOutput):
            x = df[column]
            y = df['output']
            slope, intercept = np.polyfit(x, y, 1)
            line = slope*x + intercept
            plt.scatter(x, y, marker = 'o', alpha = 0.5)
            plt.plot(x, line, color='red', alpha = 0.25)
            plt.title(f'({data_set}) - {column} vs. output')
            plt.xlabel(f'Value of {column}')
            plt.ylabel('Value of output')
        else:
            plt.scatter(df.index, df[column], marker = '1', alpha = 0.5)
            plt.title(f'Scatter Plot ({data_set}) - {column}')
            plt.xlabel('Position')
            plt.ylabel('Value')

    plt.subplots_adjust(hspace = 0.7, wspace = 0.5)

    scatter_plots_path = os.path.join(path_for_figures, figure_name)
    plt.savefig(scatter_plots_path)


def heat_map(df, data_set, path_for_figures, figure_name):
    """
    Creates and saves a heat map for the correlation matrix of the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - data_set (str): Identifier for the dataset.
    - path_for_figures (str): Path to save the figure.
    - figure_name (str): Name of the saved figure.
    """
    correlation_matrix = df.corr()

    plt.figure(figsize=(8, 8))
    heatmap = plt.imshow(correlation_matrix, cmap = 'coolwarm', vmin = -1, vmax = 1)
    plt.colorbar(heatmap)

    plt.title(f'Correlation on {data_set} data set')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation = 45, ha = "right")
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)

    heat_maps_path = os.path.join(path_for_figures, figure_name)
    plt.savefig(heat_maps_path)


def histogram(df, data_set, path_for_figures, figure_name):
    """
    Creates and saves histograms for each column in the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - data_set (str): Identifier for the dataset.
    - path_for_figures (str): Path to save the figures.
    - figure_name (str): Name of the saved figure.
    """
    plt.figure(figsize = (12, 16))

    for i, column in enumerate(df.columns[1:]):
        plt.subplot(7, 3, i + 1)
        plt.hist(df[column], bins = 33, color = 'skyblue', edgecolor = 'black', alpha = 1, label = column)
        plt.hist(df['output'], bins = 33, color = 'yellow', edgecolor = 'black', alpha = 0.5, label= 'output')
        plt.xticks([i/10 for i in range(11)])
        plt.title(f'Histogram ({data_set}) - {column}')
        plt.xlabel('Value')
        plt.ylabel('Amount of data')
        plt.legend()

    plt.subplots_adjust(hspace = 0.7, wspace = 0.3)

    histograms_path = os.path.join(path_for_figures, figure_name)
    plt.savefig(histograms_path)