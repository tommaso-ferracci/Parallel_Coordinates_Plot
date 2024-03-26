import numpy as np 
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# inspired by https://github.com/gregornickel/pcp, refactored to work with pandas dataframes
def set_ytype(data, ytype=None, colorbar=False):
    """
    Specifies for each column if the data is factor, linear or logarithmic.

    Arguments:
        data (pd.DataFrame): results dataframe from hyperparameter tuning
        ytype (list): list specifying "factor", "linear" or "log" for each column (by default None)
        colorbar (bool): True if colorbar needs to be displayed (by default False)

    Returns:
        ytype (list): list specifying "factor", "linear" or "log" for each column
    """
    if ytype == None:
        ytype = [[] for _ in range(len(data.columns))]
    for i in range(len(ytype)):
        if not ytype[i]:
            if type(data.iloc[0, i]) is str:
                ytype[i] = "factor"
            else:
                ytype[i] = "linear" 
    if colorbar: 
        assert ytype[0] == "linear", "colorbar axis needs to be linear"
    return ytype

def set_ylabels(data, ytype, ylabels=None):
    """
    Specifies for all columns the labels to use in the plot.

    Arguments:
        data (pd.DataFrame): results dataframe from hyperparameter tuning
        ytype (list): list specifying "factor", "linear" or "log" for each column (returned by set_ytype)
        ylabels (list): list specifying the correct labels for each column (by default None)

    Returns:
        ylabels (list): list specifying the correct labels for each column
    """
    if ylabels == None:
        ylabels = [[] for _ in range(len(data.columns))]
    for i in range(len(ylabels)): 
        # generate ylabels for categorical columns
        if not ylabels[i] and ytype[i] == "factor":
            ylabel = []
            for j in range(len(data)):
                if data.iloc[j, i] not in ylabel:
                    ylabel.append(data.iloc[j, i])
            ylabel.sort()
            if len(ylabel) == 1:
                ylabel.append("")
            ylabels[i] = ylabel
    return ylabels

def replace_str_values(data, ytype, ylabels):
    """
    Casts dataframe to numpy for better handling of strings in matplotlib.

    Arguments:
        data (pd.DataFrame): results dataframe from hyperparameter tuning
        ytype (list): list specifying "factor", "linear" or "log" for each column (returned by set_ytype)
        ylabels (list): list specifying the correct labels for each column (returned by set_ylabels)

    Returns:
        data (np.ndarray): processed data array
    """
    for i in range(len(ytype)):
        if ytype[i] == "factor":
            for j in range(len(data)):
                data.iloc[j, i] = ylabels[i].index(data.iloc[j, i])
    return data.to_numpy().T

def set_ylim(data, ylim=None):
    """
    Defines y-axis limits for columns in data.

    Arguments:
        data (np.ndarray): processed data array (returned by replace_str_values)
        ylim (list): list specifying y-axis limits for columns in data (by default None)

    Returns:
        ylim (list): list specifying y-axis limits for columns in data
    """
    if ylim == None:
        ylim = [[] for _ in range(data.shape[0])]
    for i in range(len(ylim)):
        if not ylim[i]:
            # default [min, max] for y-axis limits
            ylim[i] = [np.min(data[i, :]), np.max(data[i, :])]
            # if min and max are equal add some spacing
            if ylim[i][0] == ylim[i][1]:
                ylim[i] = [ylim[i][0] * 0.95, ylim[i][1] * 1.05]
            # if min and max are both 0 default to [0, 1]
            if ylim[i] == [0.0, 0.0]:
                ylim[i] = [0.0, 1.0]
    return ylim

def get_performance(data, ylim):
    """
    Retrieves and scales the last column corresponding to performance metric.

    Arguments:
        data (np.ndarray): processed data array (returned by replace_str_values)
        ylim (list): list specifying y-axis limits for columns in data (returned by set_ylim)

    Returns:
        performance (np.ndarray): performance metric scaled to [0, 1]
    """
    y_min = ylim[-1][0]
    y_max = ylim[-1][1]
    performance = (np.copy(data[-1, :]) - y_min) / (y_max - y_min)
    return performance

def rescale_data(data, ytype, ylim):
    """
    Scales secondary y-axes to scale of the main y-axis to guarantee alignment.

    Arguments:
        data (np.ndarray): processed data array (returned by replace_str_values)
        ytype (list): list specifying "factor", "linear" or "log" for each column (returned by set_ytype)
        ylim (list): list specifying y-axis limits for columns in data (returned by set_ylim)

    Returns:
        data (np.ndarray): rescaled data array
    """
    min_0 = ylim[0][0]
    max_0 = ylim[0][1]
    # main y-axis scale
    scale = max_0 - min_0
    for i in range(1, len(ylim)):
        min_i = ylim[i][0]
        max_i = ylim[i][1]
        if ytype[i] == "log":
            logmin_i = np.log10(min_i)
            logmax_i = np.log10(max_i)
            # secondary y-axis scale if data is logarithmic
            scale_i = logmax_i - logmin_i
            data[i, :] = ((np.log10(data[i, :]) - logmin_i) / scale_i) * scale + min_0
        else:
            data[i, :] = ((data[i, :] - min_i) / (max_i - min_i)) * scale + min_0
    return data

def get_path(data, i):
    """
    Interpolates a curve between the various points for a given observation.

    Arguments:
        data (np.ndarray): processed and rescaled data array (returned by rescale_data)
        i (int): specifies which observation we are considering

    Returns:
        path (matplotlib.path.Path): interpolated curve
    """
    n = data.shape[0] # number of y-axes
    verts = list(zip([x for x in np.linspace(0, n - 1, n * 3 - 2)], 
        np.repeat(data[:, i], 3)[1:-1]))
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    return path

def plot_par_coordinates(data, labels=None, ytype=None, ylim=None, ylabels=None, figsize=(10, 5), 
                         rect=[0.125, 0.1, 0.75, 0.8], curves=True, linewidth=1.0, alpha=1.0, 
                         colorbar=True, colorbar_width=0.02, cmap=plt.get_cmap("inferno_r")):
    """
    Displays a parallel coordinates plot of the hyperparameters search.

    Arguments:
        data (pd.DataFrame): pandas dataframe where the last column is the performance metric, the others represent the various 
            hyperparameters values for the search
        labels (list): labels for y-axes (default to None)
        ytype (list): data type for each axis. Defaults to None which is then mapped to "factor" for strings and "linear" for floats. 
            If ytype is passed, logarithmic axes are also possible, example: ["factor", "linear", "log", [], ...]. Empty fields must 
            be filled with an empty list []
        ylim (list): limits for each axis (default to None). Custom min and max values can be passed, example: [[0, 1], [], ...]
        ylabels (list): y-tick labels for each axis (default to None). Only use this option if you want to print more categories than you 
            have in your dataset for factor axes. Requires ylim to be set correctly
        figsize (tuple): width, height in inches of the image (default to None)
        rect: (list): [left, bottom, width, height], defines the position of the figure on the canvas (default to [0.125, 0.1, 0.75, 0.8])
        curves (bool): if True, B-spline curves are drawn instead of simple lines (default to True)
        linewidth (float): width of the curves/lines (default to 1.0)
        alpha (float): transparency of the curves/lines in [0, 1] scale (default to 1.0)
        colorbar (bool): if True, colorbar for the performance metric is drawn (default to True)
        colorbar_width (float): width of the colorbar (default to 0.02)
        cmap (matplotlib.colors.Colormap): color palette for colorbar (default to "inferno_r")
    
    Returns:
        fig (matplotlib.figure.Figure): figure to be displayed, saved or further customized
    """
    [left, bottom, width, height] = rect
    # work on copy as the dataframe will be processed
    data = data.copy(deep=True)
    # set default labels to column names
    if labels is None:
        labels = data.columns

    # processing
    ytype = set_ytype(data, ytype, colorbar) 
    ylabels = set_ylabels(data, ytype, ylabels)
    data = replace_str_values(data, ytype, ylabels)
    ylim = set_ylim(data, ylim)
    performance = get_performance(data, ylim)
    # notice: rescale_data affects only secondary y-axes
    data = rescale_data(data, ytype, ylim)

    # create figure
    fig = plt.figure(figsize=figsize)
    # create axes
    ax0 = fig.add_axes([left, bottom, width, height])
    axes = [ax0] + [ax0.twinx() for i in range(data.shape[0] - 1)]

    for i in range(data.shape[1]):
        # set color to colormap if provided
        if colorbar:
            color = cmap(performance[i])
        # otherwise default to blue
        else:
            color = "blue"

        # plot interpolated curve if requested 
        if curves:
            path = get_path(data, i)
            patch = PathPatch(path, facecolor="None", lw=linewidth, alpha=alpha, 
                    edgecolor=color, clip_on=False)
            ax0.add_patch(patch)
        # otherwise plot simple lines
        else:
            ax0.plot(data[:, i], lw=linewidth, alpha=alpha, color=color, clip_on=False)

    # format x-axis
    ax0.xaxis.tick_top()
    ax0.xaxis.set_ticks_position("none")
    ax0.set_xlim([0, data.shape[0] - 1])
    ax0.set_xticks(range(data.shape[0]))
    ax0.set_xticklabels(labels, fontsize=12)

    # format y-axis
    for i, ax in enumerate(axes):
        # reposition left spine
        ax.spines["left"].set_position(("axes", 1 / (len(labels) - 1) * i))
        # remove top, right and bottom spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        # set ylim, yscale, yticks and yticklabels
        ax.set_ylim(ylim[i])
        if ytype[i] == "log":
            ax.set_yscale("log")
        if ytype[i] == "factor":
            ax.set_yticks(range(len(ylabels[i])))
        if ylabels[i]:
            ax.set_yticklabels(ylabels[i])
        
    if colorbar:
        # add axis for colorbar and customize it
        bar = fig.add_axes([left + width, bottom, colorbar_width, height])
        norm = matplotlib.colors.Normalize(vmin=ylim[-1][0], vmax=ylim[-1][1])
        matplotlib.colorbar.ColorbarBase(bar, cmap=cmap, norm=norm, orientation="vertical")
        bar.tick_params(size=0)
        bar.set_yticklabels([])

    return fig

