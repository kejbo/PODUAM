""" Module containing functions to visualize results from UAM and standard ML models

"""

import os
import glob
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Ellipse
import matplotlib.patches as patches
import matplotlib.ticker as ticker
from rdkit import Chem
from rdkit.Chem import Draw
from scipy.stats import pearsonr

from modules.training import *


# Figure 1,2
def plot_prediction_performance(out, mg=True, lim=(-7.5, 4.5), internal=False, test_label='Test data (CV)', ax=None, fontsize=10):
    """ Creates a scatter plot of measured vs. predicted target values based on the structured
    output file from crossvalidation

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the crossvalidation
    mg: bool, default=True
        Determines whether results are plotted in mass- or molar-based units
    internal: bool, default=False
        Determines whether internal predictions are plotted alongside external crossvalidation results
    ax : matplotlib.axes.Axes, default:None
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # - calculate error metrics
    if mg:
        suffix = '_mg'
        unit = '[log(mg/kg-d)]'
        llim = lim[0]
        ulim = lim[1]
    else:
        suffix = ''
        unit = '[log(mol/kg-d)]'
        llim = -13
        ulim = -1

    y = 'y' + suffix
    yhat = 'yhat' + suffix

    mae_cv = mean_absolute_error(out[yhat], out[y])
    mdae_cv = median_absolute_error(out[yhat], out[y])
    rmse_cv = root_mean_squared_error(out[yhat], out[y])
    r2_cv = r2_score(out[yhat], out[y])
    #r2adj_cv = r2adj_score_cv(out, col_yhat=yhat, col_y=y)

    # - create scatterplot
    plt.axes(ax)
    fig = sns.scatterplot(x=yhat, y=y, data=out, s=25, color='mediumblue', linewidth=0, alpha=0.3, legend=False, ax=ax)
    sns.lineplot(x=[llim-1, ulim+1], y=[llim-1, ulim+1], color='black', alpha=0.5, linestyle=':')
    plt.xlim([llim, ulim])
    plt.ylim([llim, ulim])
    plt.xlabel('predicted POD ' + unit, fontsize=fontsize)
    plt.ylabel('reported POD ' + unit, fontsize=fontsize)
    plt.text(x=ulim - abs((ulim - llim) * 0.025), y=llim + abs((ulim - llim) * 0.05),
             ha='right', va='bottom', family='monospace',
             s='{0}\n'
               'RMSE:  {1:.3f}\n'
               'MAE:   {2:.3f}\n'
               'MdAE:  {3:.3f}\n'
               'R2:    {4:.3f}\n'
               #'R2adj: {5:.3f}\n'
               'n={5:,}'.format(test_label, rmse_cv, mae_cv, mdae_cv, r2_cv, #r2adj_cv, 
                                len(out)), size=fontsize, color='mediumblue')
    if internal:
        yhat_in = 'yhat_in' + suffix
        mae_in = mean_absolute_error(out[yhat_in], out[y])
        mdae_in = median_absolute_error(out[yhat_in], out[y])
        rmse_in = root_mean_squared_error(out[yhat_in], out[y])
        r2_in = r2_score(out[yhat_in], out[y])
        #r2adj_in = r2adj_score_cv(out, col_yhat=yhat_in, col_y=y)

        sns.scatterplot(x=yhat_in, y=y, data=out, s=25, color='teal', linewidth=0, alpha=0.3, legend=False, ax=ax)
        plt.text(x=llim + abs((ulim-llim) * 0.025), y=ulim - abs((ulim-llim) * 0.05),
                 ha='left', va='top', family='monospace',
                 s='Training data\n'
                   'RMSE:   {0:.3f}\n'
                   'MAE:    {1:.3f}\n'
                   'MdAE:   {2:.3f}\n'
                   'R2:     {3:.3f}\n'
                   #'R2adj:  {4:.3f}\n'
                   'n={4:,}'.format(rmse_in, mae_in, mdae_in, r2_in, #r2adj_in, 
                                    len(out)), size=fontsize, color='teal')

    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.axes(ax)
    plt.rcParams.update({'font.size': fontsize})

    return fig


def plot_histogram_uncertainty(out, ax, fontsize):
    """ Creates a histogram of the prediction uncertainty in terms of 95% confidence interval width
    based on the structured output file from crossvalidation

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the crossvalidation
    ax : matplotlib.axes.Axes, default:None
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    #  - plot histogram
    plt.axes(ax)
    fig = sns.histplot(out['uhat'], color='mediumblue', alpha=0.5, edgecolor='white', kde=True, stat='percent', ax=ax)
    ax.set_ylabel('fraction of chemicals [%]', fontsize=fontsize)
    ax.set_xlabel('prediction uncertainty (95% CI width)', fontsize=fontsize)
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': fontsize})

    return fig


def plot_calibration_confidence(coverage, pred_coverage, ax=None, fontsize=10):
    """ Creates a scatter plot of expected vs. observed coverage (fraction of measured values within confidence interval)

    Inputs
    ----------
    coverage : pandas series, mandatory
        The confidence levels = expected fraction of measured values in the confidence interval ("expected coverage")
    pred_coverage: pandas series, mandatory
        The observed fraction of measured values in the confidence interval ("predicted coverage")
    ax : matplotlib.axes.Axes, default:None
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # - calculate calibration metrics
    pearson = stats.pearsonr(coverage, pred_coverage).statistic
    spearman = stats.spearmanr(coverage, pred_coverage).statistic
    ece = np.abs(np.array(coverage)-np.array(pred_coverage)).mean()

    # - create scatterplot
    llim = 0
    ulim = 1

    plt.axes(ax)
    fig = sns.scatterplot(x=coverage, y=pred_coverage, s=50, color='mediumblue', linewidth=0, alpha=0.6, ax=ax)
    sns.lineplot(x=[0, ulim], y=[0, ulim], color='black', alpha=0.5, linestyle=':')
    plt.xlim([0, ulim])
    plt.ylim([0, ulim])
    plt.xlabel('expected fraction inside CI')
    plt.ylabel('observed fraction inside CI')
    plt.text(x=ulim - abs((ulim - llim) * 0.025), y=llim + abs((ulim - llim) * 0.05),
             ha='right', va='bottom', family='monospace',
             s='ECE:      {0:.3f}\n'
               'Pearson:  {1:.3f}\n'
               'Spearman: {2:.3f}'.format(ece, pearson, spearman), size=fontsize, color='mediumblue')
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': fontsize})

    return fig


def plot_calibration_error(out, col_uhat='uhat_std', num_batches='entropy', equal_width=False, ax=False, fontsize=10):
    """ Creates a scatter plot of observed prediction error vs. prediction uncertainty based on the structured
    output file from crossvalidation

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the crossvalidation
    col_uhat: str, mandatory
        The column name containing the prediction uncertainty as standard deviation (equivalent)
    num_batches: int or 'entropy', default: 'entropy'
        The number of batches that points should be grouped in
    equal_width: bool, default=False
        Determines whether points are batched based on equal number of points (default) or equal interval width
    ax : matplotlib.axes.Axes, default:None
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # - sort points by increasing prediction uncertainty
    y = out['y']
    yhat = out['yhat']
    uhat = out[col_uhat]  # needs to be STD equivalent

    z = yhat / uhat  # z-score

    sorted_uhat = pd.Series(uhat).sort_values()
    sorted_indices = sorted_uhat.index
    
    # - create batches by equal size or equal width along sorted prediction uncertainty
    if num_batches=='entropy':
        # based on Pernot (2023), http://arxiv.org/abs/2305.11905
        num_batches = np.round(len(out)**0.5)

    if not equal_width:
        idx_batches = np.array_split(sorted_indices, num_batches)
    else:
        num_batches_equal = int(np.ceil((sorted_uhat.max() - sorted_uhat.min()) / equal_width))
        num_batches = num_batches_equal
        idx_batches = [sorted_indices[(sorted_uhat >= i * equal_width) &
                                      (sorted_uhat < (i + 1) * equal_width)] for i in range(num_batches_equal)]

    
    # - calculate root mean squared error and root mean uncertainty per batch
    rmse_b = list()
    rmu_b = list()
    var_z_b = list()

    for b, bidx in enumerate(idx_batches):
        if len(bidx) != 0:
            rmse_b.append(root_mean_squared_error(yhat[bidx], y[bidx]))
            rmu_b.append((uhat[bidx] ** 2).mean() ** 0.5)
            var_z_b.append(np.var(z[bidx]))

    rmse_b = np.array(rmse_b)
    rmu_b = np.array(rmu_b)
    var_z_b = np.array(var_z_b)

    # - calculate calibration metrics
    ence = np.mean(np.abs(rmse_b - rmu_b) / rmu_b)  # MAD between RMSE and RMU
    rmse_all = root_mean_squared_error(yhat, y)
    rmu_all = (uhat ** 2).mean() ** 0.5
    pearson = stats.pearsonr(rmu_b, rmse_b).statistic
    spearman = stats.spearmanr(rmu_b, rmse_b).statistic
    
    # - create scatterplot
    llim = 0
    ulim = np.ceil(np.max([rmse_b, rmu_b]) / 0.5) * 0.5

    plt.axes(ax)
    fig = sns.scatterplot(x=rmu_b, y=rmse_b, s=50, color='mediumblue', linewidth=0, alpha=0.6, ax=ax)
    sns.lineplot(x=[0, 10], y=[0, 10], color='black', alpha=0.5, linestyle=':')
    plt.xlim([0, ulim])
    plt.ylim([0, ulim])
    plt.xlabel('mean prediction uncertainty\n(RMU per batch)')
    plt.ylabel('observed prediction error\n(RMSE per batch)')
    plt.text(x=ulim - abs((ulim - llim) * 0.025), y=llim + abs((ulim - llim) * 0.05),
             ha='right', va='bottom', family='monospace',
             s='RMU:      {0:.3f}\n'
               'RMSE:     {1:.3f}\n'
               'ENCE:     {2:.3f}\n'
               'Pearson:  {3:.3f}\n'
               'Spearman: {4:.3f}\n'
               'n={5:,}, batches={6:.0f}'.format(rmu_all, rmse_all, ence, pearson, spearman,
                                                   len(out), num_batches), size=fontsize, color='mediumblue')
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': fontsize})

    return fig, ence, num_batches


def plot_calibration_distance(out, col_uhat='uhat_std', col_dJ='dJ', res=False, num_batches='entropy', equal_width=False, ax=None, fontsize=10):
    """ Creates a scatter plot of the average Jaccard distance vs. prediction uncertainty based on the structured
    output file from crossvalidation

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the crossvalidation
    col_uhat: str, mandatory, default: 'uhat_std'
        The column name containing the prediction uncertainty
    col_dJ: str, mandatory, default: 'dJ'
        The column name containing the Jaccard distances
    res: bool, default: False
        Option to make the plot based on the residuals ("observed uncertainty") instead of the prediction uncertainty. 
        In this case, the column name of the residuals needs to be passed into col_uhat.
    num_batches: int or 'entropy', default: 'entropy'
        The number of batches that points should be grouped in
    equal_width: bool, default=False
        Determines whether points are batched based on equal number of points (default) or equal interval width
    ax : matplotlib.axes.Axes, default:None
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # - sort points by increasing prediction uncertainty
    dJ_test = out[col_dJ]
    uhat = out[col_uhat]

    sorted_uhat = pd.Series(uhat).sort_values()
    sorted_indices = pd.Series(uhat).sort_values().index

    # - create batches by equal size or equal width along sorted prediction uncertainty
    if num_batches=='entropy':
        # based on Pernot (2023), http://arxiv.org/abs/2305.11905
        num_batches = np.round(len(out)**0.5)
    
    if not equal_width:
        idx_batches = np.array_split(sorted_indices, num_batches)
    else:
        num_batches_equal = int(np.ceil((sorted_uhat.max() - sorted_uhat.min()) / equal_width))
        num_batches = num_batches_equal
        idx_batches = [sorted_indices[(sorted_uhat >= i * equal_width) &
                                      (sorted_uhat < (i + 1) * equal_width)] for i in range(num_batches_equal)]

    # - calculate mean Jaccard distance and root mean uncertainty per batch
    dJ_b = list()
    rmu_b = list()

    for b, bidx in enumerate(idx_batches):
        dJ_b.append(dJ_test[bidx].mean())
        rmu_b.append((uhat[bidx] ** 2).mean() ** 0.5)

    dJ_b = np.array(dJ_b)
    rmu_b = np.array(rmu_b)

    # - calculate calibration metrics
    dJ_all = dJ_test.mean()
    rmu_all = (uhat ** 2).mean() ** 0.5
    pearson = stats.pearsonr(rmu_b, dJ_b).statistic
    spearman = stats.spearmanr(rmu_b, dJ_b).statistic

    # - create scatterplot
    llim = 0
    ulimx = np.ceil(max(rmu_b) / 0.5) * 0.5
    ulimy = 1

    plt.axes(ax)
    fig = sns.scatterplot(x=rmu_b, y=dJ_b, s=50, color='mediumblue', linewidth=0, alpha=0.6, ax=ax)
    plt.xlim([0, ulimx])
    plt.ylim([0, ulimy])
    if res:
        plt.xlabel('observed prediction error (RMSE per batch)')
    else:
        plt.xlabel('mean prediction uncertainty\n(RMU per batch)')
    plt.ylabel('mean chemical dissimilarity\n(Jaccard distance per batch)')
    plt.text(x=ulimx - abs((ulimx - llim) * 0.025), y=llim + abs((ulimy - llim) * 0.05),
             ha='right', va='bottom', family='monospace',
             s='RMU:      {0:.3f}\n'
               'Jaccard:  {1:.3f}\n'
               'Pearson:  {2:.3f}\n'
               'Spearman: {3:.3f}\n'
               'n={4:,}, batches={5:.0f}'.format(rmu_all, dJ_all, pearson, spearman, len(out), num_batches),
             size=fontsize, color='mediumblue')
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': fontsize})

    return fig


# Figure 3
def plot_chemspace(out, hue_col='', CI=False, add_pie=True, legend=True, annotate=True, clusters=None, prefix='', fontsize=10, ax=None):
    """ Creates a 2-dimensional spatial map of all marketed chemicals based on t-SNE coordinates and colors each point
    by predicted toxicity or prediction uncertainty

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the application of the final models
    hue_col: str, mandatory
        The column name of the variable used for the hue: 'yhat_mg' or 'uhat'
    CI: int or False, default: False
        The type of uncertainty used for the legend. False indicates a standard deviation, an integer defines the confidence interval in %. 
    add_pie: bool, default: True
        Option to add a pie chart summarizing the distribution of the hue variable
    legend: bool, default: True
        Option to add a legend for the hue variable
    annotate: bool, default: True
        Option to add annotations provided through "clusters"
    clusters: dict, default: None
        The dictionary defining the location and size of the clusters to annotate
    prefix: str, default: ''
        Prefix to use in front of the cluster number, optional.
    ax : matplotlib.axes.Axes, default:None
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # - define the bin sizes, palette and legend for hue
    if 'uhat' in hue_col:
        if CI==90:
            bins = pd.IntervalIndex.from_tuples([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 10)])
        elif CI==95:
            bins = pd.IntervalIndex.from_tuples([(0, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 10)])
        else:
            raise ValueError("Specify bins for {CI}% confidence interval".format(CI=CI))
        palette = sns.diverging_palette(250, 10, s=100, l=50, sep=1, n=len(bins), center='light')
        legend_title = 'uncertainty\n{CI}% CI width'.format(CI=CI)
    elif ('yhat' in hue_col) & ('mg' in hue_col):
        bins = pd.IntervalIndex.from_tuples([(-10, 0), (0, 0.5), (0.5, 1), (1, 1.5), (1.5, 2), (2, 10)])
        palette = sns.diverging_palette(30, 210, s=100, l=50, sep=1, n=len(bins), center='light')
        legend_title = 'toxicity (POD)\n[log(mg/kg-d)]'
    else:
        raise ValueError('Choose uhat or yhat_mg as hue color. Define new bins otherwise.')

    hue_bins = pd.cut(out[hue_col], bins)

    # - create chemical space plot
    plt.axes(ax)
    fig = sns.scatterplot(data=out, x='TSNE1', y='TSNE2', hue=hue_bins, palette=palette,
                          s=5, alpha=0.3, linewidth=0, legend=legend, ax=ax)
    ax.axis('off')


    if legend:
        handles, labels = ax.get_legend_handles_labels()
        if 'yhat' in hue_col:  # invert legend for yhat
            handles.reverse()
            labels.reverse()

        leg = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.1, 0.9), title=legend_title)
        for lh in leg.legendHandles:
            lh.set_alpha(1)
            lh._markersize = 5

    plt.rcParams.update({'font.size': fontsize})

    # - add pie chart
    if add_pie:
        if 'yhat' in hue_col:
            ascending, palette_pie = False, palette[::-1]
        else:
            ascending, palette_pie = True, palette

        freq_data = hue_bins.value_counts().sort_index(ascending=ascending)
        ax_pie = inset_axes(ax,
                            width=1.7,
                            height=1.7,
                            bbox_transform=ax.transAxes,
                            bbox_to_anchor=(0.87, 0.3),
                            loc='upper left')
        ax_pie.pie(freq_data, colors=palette_pie, autopct='%.0f%%', pctdistance=1.2)
        ax_pie.axis('off')

    # - add cluster annotations
    if annotate:
        if not clusters:
            raise ValueError('Provide cluster dictionary or set "annotate" to False.')

        for cnum in np.arange(len(clusters['group_name'])):

            xm, ym = clusters['xy'][cnum]
            xd, yd = clusters['width'][cnum], clusters['height'][cnum]
            angle = clusters['angle'][cnum]

            ellipse = Ellipse(xy=(xm, ym), width=xd, height=yd, angle=angle, edgecolor='black', fc='None', lw=1.5)
            ax.add_patch(ellipse)
            ax.text(xm + np.sign(angle)*(xd/4), ym - yd / 2 - 2, prefix + str(cnum + 1), fontsize=fontsize + 2, fontweight='bold',
                    ha='center', va='top')
    return fig


def append_values_to_keys(dictionary, key_value_pairs):
    """ Support function to add entries to a dictionary from a list of tuples

    Inputs
    ----------
    dictionary : dict, mandatory
        The dictionary to add entries to
    key_value_pairs: list, mandatory
        The list of tuples containing the key and corresponding value

    """

    for key, value in key_value_pairs:
        if key in dictionary:
            dictionary[key].append(value)
        else:
            dictionary[key] = [value]


def plot_annotated_clusters(clusters, ax, num_per_row=5, prefix='', fontsize=10):
    """ Creates legend for annotated clusters with the chemical class name and a representative chemical structure

    Inputs
    ----------
    clusters: dict, default: None
        The dictionary defining the chemical class and representative chemical structure for each annotated clusters
    num_per_row: int, default: 6
        The number of clusters described per row
    prefix: str, default: ''
        Prefix to use in front of the cluster number, optional
    ax : matplotlib.axes.Axes, mandatory
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    """

    ax.axis('off')
    row_count = 0
    for cnum in np.arange(len(clusters['group_name'])):
        m = Chem.MolFromSmiles(clusters['smiles'][cnum])
        Chem.rdCoordGen.AddCoords(m)
        im_arr = Draw.MolToImage(m, size=(300, 300), fitImage=True)

        pad = 1.05/num_per_row-0.1
        ax_cluster = inset_axes(ax, width=0.8, height=0.8, bbox_transform=ax.transAxes,
                            bbox_to_anchor=(pad + (pad+0.1) * (cnum % num_per_row), 0.7 - 0.38 * row_count), loc='upper left')
        ax_cluster.imshow(im_arr)
        ax_cluster.text(-0.2, 1.3, prefix + str(cnum + 1), ha='right', va='top', fontsize=fontsize+2,
                    fontweight='bold', transform=ax_cluster.transAxes)
        ax_cluster.text(0, 1, clusters['group_name'][cnum],
                    ha='left', va='bottom', fontsize=fontsize, transform=ax_cluster.transAxes)
        ax_cluster.axis('off')

        if (cnum + 1) % num_per_row == 0:
            row_count += 1


# Figure 4
def plot_group_ranking_scatter(out, groupby='my_class', col_uhat='uhat_std', CI=False, legend=False, highlight_top=50,
                               min_chem=10, gradient=False, fontsize=10, ax=None, **kwargs):
    """ Creates a scatterplot of the predicted POD vs. the prediction uncertainty for all chemical classes among marketed chemicals

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the application of the final models
    groupby: str, default: 'my_class'
        The chemical class level by which to group chemicals
    col_uhat: str, mandatory
        The column name of the prediction uncertainty
    CI: int or False, default: False
        The type of uncertainty used for the label. False indicates a standard deviation, an integer defines the confidence interval in %. 
    legend: bool, default: True
        Option to add a legend for the hue variable
    highlight_top: int, default: 50
        The number of chemical classes with highest predicted toxicity to highlight
    min_chem: int, default: 10
        The number of minimum chemicals in a given chemical classes to be considered for highlight_top
    ax: matplotlib.axes.Axes, mandatory
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends
    **kwargs: optional
        Any additional arguments to be passed to sns.scatterplot

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    #  - group chemicals
    out_grouped = out.groupby(['Superclass (top 15)', groupby], as_index=False)[['yhat_mg', 'uhat']].median()

    #  - plot scatterplot
    xmin, xmax = -3.5, 3
    ymin, ymax = 0, 10
    
    plt.axes(ax)
    fig = sns.scatterplot(out_grouped, x='yhat_mg', y=col_uhat, s=20, linewidth=0.1, alpha=0.7, ax=ax, legend=legend,
                          **kwargs)
    ax.set_xlabel('POD [log(mg/kg-d)]')
    ax.set_ylabel('uncertainty ({utype})'.format(utype='std dev' if not CI else '{CI}% CI width'.format(CI=CI)))
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if legend:
        leg = ax.legend(ncol=4, loc="lower left", bbox_to_anchor=(-0.05, 1.05), title='Superclass (top 15)')
        for lh in leg.legendHandles:
            lh.set_alpha(1)
    
    # - plot background gradient
    if gradient==True:
        
        x = np.linspace(1, 0, 256)
        y = np.linspace(1, 0, 256)
        xArray, yArray = np.meshgrid(x, y)
        plotArray = np.sqrt(xArray ** 2 + yArray ** 2)
        ax.imshow(plotArray, cmap='coolwarm', vmin=0, vmax=1, extent=(xmax, xmin, ymin, ymax), alpha=0.2, aspect='auto')
    
   # - highlight area with top 50 most toxic chemical groups
    if highlight_top > 0:
        data_agg = out.groupby(groupby)['yhat_mg'].agg(['count', 'median']).sort_values(by='median')
        data_agg_top = data_agg[data_agg['count'] >= min_chem]['median'].sort_values()[:highlight_top]

        zoom_xlim = (out_grouped['yhat_mg'].min()-0.1, data_agg_top.iloc[highlight_top-1])
        ax.axvline(zoom_xlim[0], color='black', linestyle='--', linewidth=2)
        ax.axvline(zoom_xlim[1], color='black', linestyle='--', linewidth=2)
        ax.fill_betweenx(ax.get_ylim(), zoom_xlim[0], zoom_xlim[1], color='grey', alpha=0.1)

    plt.rcParams.update({'font.size': fontsize})

    return fig


def plot_group_counts(data, agg_col='yhat_mg', groupby='my_class', highlight_top=50, min_chem=30, fontsize=10, ax=None,
                      **kwargs):
    """ Creates a barplot of the number of chemicals per chemical class for the most toxic chemical classes

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the application of the final models
    groupby: str, default: 'my_class'
        The chemical class level by which to group chemicals
    highlight_top: int, default: 50
        The number of chemical classes with highest predicted toxicity to include
    min_chem: int, default: 10
        The number of minimum chemicals in a given chemical classes to be considered for highlight_top
    ax: matplotlib.axes.Axes, mandatory
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends
    **kwargs: optional
        Any additional arguments to be passed to sns.barplot

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    #  - group chemicals
    data_agg = data.groupby(['Superclass (top 15)', groupby])['yhat_mg'].agg(['count', 'median']).sort_values(by='median')
    data_agg_top = data_agg[data_agg['count'] >= min_chem].sort_values(by='median')[:highlight_top].reset_index()
    
    #  - plot barplot
    plt.axes(ax)
    fig = sns.barplot(x=groupby, y='count', data=data_agg_top, legend=False, color='black', alpha=0.5, width=0.5, ax=ax,
                      **kwargs)
  
    ax.grid(axis='y', which='both', color='lightgrey', linestyle='-', linewidth=0.5)

    ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(250))

    ax.set_axisbelow(True)
    ax.tick_params('x', labelrotation=90)
    ax.set_ylabel('nr of chemicals')
    plt.rcParams.update({'font.size': fontsize})

    return fig


def plot_group_ranking_violins(data, y_col='yhat_mg', y_label='POD [log(mg/kg-d)]',groupby='my_class', highlight_top=50, 
                               min_chem=30, invert_y=False, fontsize=10, ax=None, **kwargs):

    """ Creates a combined violin and boxplot chemical distribution per chemical class for the most toxic chemical classes

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the application of the final models
    groupby: str, default: 'my_class'
        The chemical class level by which to group chemicals
    highlight_top: int, default: 50
        The number of chemical classes with highest predicted toxicity to include
    min_chem: int, default: 10
        The number of minimum chemicals in a given chemical classes to be considered for highlight_top
    invert_y: bool, default: False
        Option to invert the y-axis to plot low PODs (high toxicity) at the top
    ax: matplotlib.axes.Axes, mandatory
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends
    **kwargs: optional
        Any additional arguments to be passed to sns.barplot

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    
    #  - group chemicals
    data_agg = data.groupby(groupby)['yhat_mg'].agg(['count', 'median', 'min', 'max', 'std']).sort_values(by='median')
    top_list = list(data_agg[data_agg['count'] >= min_chem]['median'].sort_values()[:highlight_top].index)
    data['my_class_top'] = data['my_class'].where(data['my_class'].isin(top_list), other=np.nan)

    #  - plot violin and barplots
    plt.axes(ax)
    fig = sns.violinplot(data=data, x='my_class_top', y=y_col, order=top_list, legend=False,
                         alpha=0.5, width=0.9, linewidth=0, color='mediumblue', density_norm='width', ax=ax,
                         **kwargs)
    sns.boxplot(data=data, x='my_class_top', y=y_col, order=top_list, legend=False,
                color='black', width=0.6, fill=False, linewidth=0.5, fliersize=1, whis=[2.5, 97.5], ax=ax)
    ax.tick_params('x', labelrotation=90)
    ax.set_xlabel('')
    ax.set_ylabel(y_label)

    ax.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    if invert_y:
        ax.invert_yaxis()

    plt.rcParams.update({'font.size': fontsize})

    return fig


# SI - Supplemental Figures
def plot_histogram_pod(data, col_x='POD', xlabel='reported POD [log(mg/kg-d)]', x_bin=1, invert=True, simple=False, 
                       add_zoom=True, ax=None, fontsize=10, **kwargs):

    """ Creates a histogram of PODs with optional highlights and top 3 most toxic chemicals

    Inputs
    ----------
    data : pandas dataframe, mandatory
        The dataframe containing the POD values (and chemical names)
    col_x: str, default: 'POD'
        The column name containing the POD values
    x_label: str, default: 'reported POD (log10 mg/kg-d)'
        The x-axis label
    x_bin: int, default:1
        The bin width passed to "binwidth" argument in sns.histplot
    invert_y: bool, default: False
        Option to invert the y-axis to plot low PODs (high toxicity) at the top
    simple: bool, default: False
        Option to remove axes
    add_zoom: bool, default: True
        Option to zoom in on the lower POD range    
    ax: matplotlib.axes.Axes, mandatory
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends
    **kwargs: optional
        Any additional arguments to be passed to sns.histplot

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    #  - plot histogram
    plt.axes(ax)
    fig = sns.histplot(data, x=col_x, binwidth=x_bin, ax=ax, color='black', edgecolor='white', multiple='stack',
                       legend=False, **kwargs)

    if invert:
        ax.invert_xaxis()
        top3_names = list(data.sort_values(by=col_x, ascending=True)['name'].iloc[:3])
        top3_values = list(data.sort_values(by=col_x, ascending=True)[col_x].iloc[:3])
        zoom_xlim = (-2, -8)
        bbox_to_anchor_zoom = (0.55, 0.1, 0.35, 0.35)
        x_text_zoom = 0.9
        ha_text_zoom = 'right'
    else:
        top3_names = list(data.sort_values(by=col_x, ascending=True)['name'].iloc[:3])
        top3_values = list(data.sort_values(by=col_x, ascending=True)[col_x].iloc[:3])
        zoom_xlim = (-8, -2)
        bbox_to_anchor_zoom = (0.1, 0.1, 0.35, 0.35)
        x_text_zoom = 0.1
        ha_text_zoom = 'left'

    if simple:
        ax.axis('off')
    else:
        ax.set_xlabel(xlabel)
        ax.set_ylabel('nr of chemicals')

    #  - add zoom on lower POD range
    if add_zoom:
        plt.text(x=x_text_zoom, y=0.95, transform=ax.transAxes,
                 ha=ha_text_zoom, va='top', family='monospace',
                 s=r'$\bf{Top\ 3\ most\ toxic\ chemicals}$' + '\n' +
                   '{name1} {value1:.2f}\n'
                   '{name2} {value2:.2f}\n'
                   '{name3} {value3:.2f}\n'.format(name1=top3_names[0], name2=top3_names[1], name3=top3_names[2],
                                                   value1=top3_values[0], value2=top3_values[1], value3=top3_values[2]),
                 size=fontsize)

        ax_zoom = inset_axes(ax, width='100%', height='100%', loc='lower left', bbox_to_anchor=bbox_to_anchor_zoom,
                             bbox_transform=ax.transAxes)
        sns.histplot(data, x=col_x, ax=ax_zoom, color='black', edgecolor='white', multiple='stack', legend=False,
                     **kwargs)
        ax_zoom.set_xlim(zoom_xlim)
        ax_zoom.set_ylim((0, 20))
        ax_zoom.set_xlabel('')
        ax_zoom.set_ylabel('')
        ax_zoom.set_yticks([0, 20])
        ax_zoom.yaxis.set_minor_locator(plt.MultipleLocator(1))

        ax.axvline(zoom_xlim[0], color='r', linestyle='--', linewidth=1)
        ax.axvline(zoom_xlim[1], color='r', linestyle='--', linewidth=1)
        old_lim = ax.get_ylim()
        ax.fill_betweenx((0, 2000), zoom_xlim[0], zoom_xlim[1], color='red', alpha=0.1)
        ax.set_ylim(old_lim)

    plt.rcParams.update({'font.size': fontsize})

    return fig


def plot_ence(out, batches, ax=None, fontsize=10):
    """ Creates a scatter plot showing how the ENCE changes with the choice of number of batches

    Inputs
    ----------
    out : pandas dataframe, mandatory
        The structured output from the crossvalidation
    batches: list of integers or 'entropy'
        List of number of batches that points should be grouped in
    ax : matplotlib.axes.Axes, default:None
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # - sort by increasing prediction uncertainty
    y = out['y']
    yhat = out['yhat']
    uhat = out['uhat_std']

    sorted_uhat = pd.Series(uhat).sort_values()
    sorted_indices = sorted_uhat.index
    
    # - calculate ENCE for different number of batches
    ence_all = list()
    n_sqrt = list()

    for num_batches in batches:
        # - create batches by equal size along sorted prediction uncertainty
        if num_batches=='entropy':
            # based on Pernot (2023), http://arxiv.org/abs/2305.11905
            num_batches = np.round(len(out)**0.5)

        idx_batches = np.array_split(sorted_indices, num_batches)
       
        # - calculate root mean squared error and root mean uncertainty per batch
        rmse_b = list()
        rmu_b = list()

        for b, bidx in enumerate(idx_batches):
            if len(bidx) != 0:
                rmse_b.append(root_mean_squared_error(yhat[bidx], y[bidx]))
                rmu_b.append((uhat[bidx] ** 2).mean() ** 0.5)

        rmse_b = np.array(rmse_b)
        rmu_b = np.array(rmu_b)

        # - calculate calibration metrics
        ence = np.mean(np.abs(rmse_b - rmu_b) / rmu_b)  # MAD between RMSE and RMU
        
        ence_all.append(ence)
        n_sqrt.append(num_batches ** 0.5)

    # - create scatterplot
    plt.axes(ax)
    fig = sns.scatterplot(x=n_sqrt, y=ence_all, s=40, color='mediumblue', linewidth=0, alpha=0.6, ax=ax)
    ax.set_ylim([0, 0.5])
    ax.set_xlabel('sqrt of number of batches')
    ax.set_ylabel('ENCE')
    plt.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': fontsize})

    return fig


def plot_histogram_with_cumulative(out_cv, out_app, col_x, x_label='x', sety=(True,True), set_label=('Test set (CV)', 'Application set'), 
                                   colors=('mediumblue', 'black'), text_pos='left', ax=None, fontsize=10):
    """ Creates a histogram of a given variable comparating the distributions for training chemicals (using external CV predictions) 
    and application chemicals incl. cumulative curve 

    Inputs
    ----------
    out_cv : pandas dataframe, mandatory
        The structured output file from the crossvalidation
    out_app : pandas dataframe, mandatory
        The structured output file from applying the final models to a set of application chemicals
    col_x: str, mandatory
        The column name containing the variable for the histogram
    x_label: str, default: 'reported POD (log10 mg/kg-d)'
        The x-axis label
    sety: tuple, default: (True, True)
    set_label: tuple, default: ('Test set (CV)', 'Application set')
        This tuple provides the labels of the two sets provided.
    colors: tuple, default: ('mediumblue', 'black')
        Colors to use for the two sets.
    text_pos: tuple, default ('top', 'left')
        This tuple defines the position of the description text.
    ax: matplotlib.axes.Axes, mandatory
        pre-existing axes for the plot
    fontsize: int, default:10
        fontsize for all axes and legends

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # - plot histogram
    sns.histplot(out_app[col_x], bins=50, color=colors[1], alpha=0.6, edgecolor='white', kde=False, stat='percent', ax=ax)
    sns.histplot(out_cv[col_x], bins=50, color=colors[0], alpha=0.6, edgecolor='white', kde=False, stat='percent', ax=ax)
    ax.set_xlabel(x_label)
    if sety[0]:
        ax.set_ylabel('Fraction of chemicals [%]')
    else:
        ax.set_ylabel('')
    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_ylim(0, 20)

    x_pos = 0.05 if text_pos=='left' else 0.95
    

    ax.text(x=x_pos, y=0.95, ha=text_pos, va='top', family='monospace', transform=ax.transAxes,
            s='{0}\n'
              'mean: {1:.3f}\n'
              'n:    {2:d}'.format(set_label[0], out_cv[col_x].mean(), len(out_cv)), size=fontsize, color=colors[0])
    ax.text(x=x_pos, y=0.75, ha=text_pos, va='top', family='monospace', transform=ax.transAxes,
            s='{0}\n'
              'mean: {1:.3f}\n'
              'n:    {2:d}'.format(set_label[1], out_app[col_x].mean(), len(out_app)), size=fontsize, color=colors[1])

    # - calculate the cumulative percentage
    n_cv, bins_cv, patches_cv = ax.hist(out_cv[col_x], bins=30, color='black', alpha=0.0)  # Get bins
    cdf_cv = np.cumsum(n_cv) / np.sum(n_cv) * 100

    n_app, bins_app, patches_app = ax.hist(out_app[col_x], bins=30, color='black', alpha=0.0)  # Get bins
    cdf_app= np.cumsum(n_app) / np.sum(n_app) * 100

    # - plot cumulative distribution on second y-axis
    ax2 = ax.twinx()
    ax2.plot(bins_cv[:-1], cdf_cv, color=colors[0], linewidth=2)
    ax2.plot(bins_app[:-1], cdf_app, color=colors[1], linewidth=2)
    if sety[1]:
        ax2.set_ylabel('Cumulative fraction [%]')
    else:
        ax2.set_ylabel('')
    ax2.set_ylim(0, 100)  # Ensure the y-axis is from 0 to 100 percent

    ax.grid(axis='y', color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    plt.rcParams.update({'font.size': fontsize})

    return ax


def load_uam_data(filepath, effect):
    """ Loads the UAM data for a given effect from the specified filepath and returns the yhat, uhat, and ytrue matrices.

    Inputs
    ----------
    filepath : str, mandatory
        The path to the directory containing the UAM data files.
    effect : str, mandatory
        The specific effect to load data for.

    Outputs
    ----------
    yhat_matrix: pandas DataFrame
        Matrix containing the predicted values (yhat) for each UAM in different columns.
    yhat_matrix: pandas DataFrame
        Matrix containing the prediction uncertainty as 95% CI width (uhat) for each UAM in different columns.
    ytrue: pandas Series
        Series containing the true values (y_mg)
    """

    pattern = os.path.join(filepath, "out_cv_*_95_*_{effect}-std.csv".format(effect=effect))

    # - rename map
    rename_map = {
        ('bnn', 'cddd'): 'BNN-CDDD',
        ('bnn', 'maccs'): 'BNN-MACCS',
        ('bnn', 'morgan'): 'BNN-Morgan',
        ('bnn', 'rdkit'): 'BNN-RDKit',
        ('cp-RF', 'cddd'): 'CP-CDDD',
        ('cp-RF', 'maccs'): 'CP-MACCS',
        ('cp-RF', 'morgan'): 'CP-Morgan',
        ('cp-RF', 'rdkit'): 'CP-RDKit'}

    # - load data
    model_yhat = {}
    model_uhat = {}
    model_ytrue = None

    for file in glob.glob(pattern):
        filename = os.path.basename(file)
        try:
            parts = filename.split('_')
            family = parts[2]
            descriptor = parts[4].split('-')[0]
            key = (family, descriptor)
            model_name = rename_map.get(key, f"{family}_{descriptor}")

            df = pd.read_csv(file)[['ID', 'yhat_mg', 'uhat', 'y_mg']]
            df = df.sort_values('ID').reset_index(drop=True)

            model_yhat[model_name] = df[['ID', 'yhat_mg']].rename(columns={'yhat_mg': model_name})
            model_uhat[model_name] = df[['ID', 'uhat']].rename(columns={'uhat': model_name})
            
            if model_ytrue is None:
                model_ytrue = df[['ID', 'y_mg']].rename(columns={'y_mg': 'y_mg'})
            else:
                assert (model_ytrue['ID'].values == df['ID'].values).all()

        except Exception as e:
            print(f"Skipping {filename}: {e}")

    # - combine all
    yhat_df = model_ytrue[['ID']].copy()
    uhat_df = model_ytrue[['ID']].copy()

    for name, df in model_yhat.items():
        yhat_df = yhat_df.merge(df, on='ID')

    for name, df in model_uhat.items():
        uhat_df = uhat_df.merge(df, on='ID')

    # - drop IDs
    yhat_matrix = yhat_df.drop(columns='ID')
    uhat_matrix = uhat_df.drop(columns='ID')
    ytrue = model_ytrue['y_mg']

    return yhat_matrix, uhat_matrix, ytrue


def plot_correlation_scatter(x, y, lim=(-4,4), **kwargs):
    """ Creates a scatter plot of two variables with a diagonal line indicating perfect correlation.

    Inputs
    ----------
    x : array-like, mandatory
    y : array-like, mandatory
    lim : tuple, default: (-4, 4)
        The limits for the x and y axes.
    **kwargs : keyword arguments
        Additional arguments passed to the scatter plot.

    Outputs
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the scatter plot.
    """
    ax = plt.gca()
    ax.scatter(x, y, color='mediumblue', alpha=0.6, s=5)
    min_val = lim[0]
    max_val = lim[1]
    ax.plot([min_val, max_val], [min_val, max_val], color='gray', linestyle='--', linewidth=1)
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6)


def plot_correlation_scatter_pairgrid(matrix, title, lim=(-4,4)):
    """ Creates a pair grid scatter plot for the pairwise correlation between predictions from different UAMs.

    Inputs
    ----------
    matrix : pandas DataFrame, mandatory 
        The matrix of prediction medians or uncertainty for different UAMs from load_uam_data(). 
    title : str, mandatory
        The title for the plot.
    lim : tuple, default: (-4, 4)
        The limits for the x and y axes.
    Outputs
    ----------
    g : seaborn.PairGrid
        The PairGrid object containing the pairwise correlation scatter plots.
    """

    g = sns.PairGrid(matrix, corner=False)
    g.map_lower(plot_correlation_scatter, lim=lim)

    # Remove upper triangle axes entirely
    for i in range(len(matrix.columns)):
        for j in range(i , len(matrix.columns)):
            ax = g.axes[i, j]
            if ax is not None:
                ax.remove()

    g.fig.text(0.5, 0.96, title, ha='center', fontsize=12, style='italic')
    g.fig.subplots_adjust(top=1.1, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    
    return g


def plot_uncertainty_variability_correlation(data, ax=None, fontsize=10):
    """ Creates a scatter plot showing the correlation between variability in median predictions across different UAMs(95% CI width) 
    and mean prediction uncertainty (95% CI width).

    Inputs
    ----------
    data : pandas DataFrame, mandatory
        The input data containing prediction variability and mean prediction uncertainty.

    Outputs
    ----------
    ax : matplotlib.axes.Axes
        The axes containing the scatter plot.
    """

    r_var, _ = pearsonr(data['prediction_ci'], data['uncertainty_mean'])
    r_err, _ = pearsonr(data['residual_abs'], data['uncertainty_mean'])

    r2_var = r_var**2
    r2_err = r_err**2

    sns.regplot(x='prediction_ci', y='uncertainty_mean', data=data, scatter_kws={'alpha':0.6, 's':5, 'color':'mediumblue'}, ax=ax)
    ax.set_xlabel('Variability (95% CI width) across median predictions')
    ax.set_ylabel('Mean prediction uncertainty (95% CI width)')
    ax.text(0.95, 0.05, f'$Pearson^2$ = {r2_var:.2f}', ha='right', va='bottom',transform=ax.transAxes, color='mediumblue', fontsize=12)

    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': fontsize})

    return ax


def collect_feature_importance(model, desc):
    """ Collects feature importance underlying quantile regression forests (qRF) inside CP models and returns a DataFrame with mean feature importances.
    Inputs 
    ----------
    model : object, mandatory
        The CP model object containing the feature importances from underlying qRFs.
    desc : list, mandatory
        List of feature names.
    Outputs
    ----------
    importance_df : pandas DataFrame
        DataFrame containing the feature importances for each feature across all underlying qRFs.
    sorted_features : list
        List of feature names sorted by their mean importance in descending order.
    """

    importances = []

    for i in range(1000):
        base_model = model.models_B[i]
        importances.append(base_model.feature_importances_)

    importance_df = pd.DataFrame(importances)
    importance_df.columns = desc

    # - rank by mean importance
    mean_importance = importance_df.mean().sort_values(ascending=False)
    sorted_features = mean_importance.index.tolist()
    importance_df = importance_df[sorted_features]

    return importance_df, sorted_features


def plot_feature_importance(data, sorted_features, ax, top_n=None, title=''):
    """ Creates a boxplot of feature importance for the top N features.
    Inputs
    ----------
    data : pandas DataFrame, mandatory
        DataFrame containing feature importance values for each feature from collect_feature_importance()
    sorted_features : list, mandatory
        List of feature names sorted by their mean importance in descending order from collect_feature_importance()
    top_n : int or None, default: None
        The number of top features to plot. If None, all features are plotted.
    title : str, default: ''
        The title for the plot.
    Outputs
    ---------- 
    ax : matplotlib.axes.Axes
        The axes containing the boxplot of feature importance.
    """

    # - filter top features
    if top_n:
        features_to_plot = sorted_features[:top_n]
        data = data[data['feature'].isin(features_to_plot)]

    # - create boxplot
    sns.boxplot(data=data,x='importance',y='feature',orient='h',ax=ax,
        fliersize=2,  linewidth=0.75, whis=[2.5, 97.5], color='mediumblue')

    ax.set_title(title)
    ax.set_xlabel("Feature importance")
    ax.set_ylabel("")
    ax.grid(True)

    return ax


def nested_cv_performance_comparison(df_metrics_fold, metric, bench, limits, ax=False, fontsize=12):
    """ Creates a combined violin and boxplot to compare the distribution of a given metric across crossvalidation folds to compare
    the prediction performance across models

    Inputs
    ----------
    df_metrics_fold : pandas dataframe, mandatory
        The structured output file from create_consensus.py containing error metrics for different models in every crossvalidation fold
    metric: str, mandatory
        The name of the error metric to plot, options: 'MAE', 'MdAE', 'RMSE', 'R2'
    bench : pandas series, mandatory
        The benchmark values for each error metric representing the performance of a mean-predicting model
    limits: tuple, mandatory
        The lower and upper limits of the y-axis

    Outputs
    ----------
    fig: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """

    # - plot boxplot
    order = ['consensus', 'CDDD embedding', 'RDKIT descriptors', 'MACCS fingerprint', 'MORGAN fingerprint']
    hue_order = ['consensus', 'XGB', 'SVM', 'RF', 'NN', 'KNN', 'MLR']
    palette_dict = dict(consensus='grey',
                        SVM='gold', KNN='coral',
                        XGB='mediumorchid', RF='teal',
                        NN='mediumblue', MLR='cornflowerblue')
    palette = [palette_dict[key] for key in hue_order]

    llim, ulim = limits

    #fig, ax = plt.subplots(figsize=(15, 8))
    #plt.subplots_adjust(bottom=0.3)

    plt.axes(ax)
    fig = sns.violinplot(x='Features', y=metric, hue='Algorithm', data=df_metrics_fold[df_metrics_fold['Algorithm']!='consensus'],
                       order=order, hue_order=hue_order, palette=palette, alpha=0.2,
                       width=0.9, dodge='auto', fill=True, linewidth=0, inner=None, legend=False, ax=ax)
    sns.boxplot(x='Features', y=metric, hue='Algorithm', data=df_metrics_fold[df_metrics_fold['Algorithm']!='consensus'],
                    order=order, hue_order=hue_order, palette=palette,
                    width=0.9, gap=0.4, dodge='auto', fill=False, linewidth=1.5, fliersize=1, whis=[2.5, 97.5], ax=ax)
    sns.violinplot(x='Features', y=metric, hue='Algorithm',
                   data=df_metrics_fold[df_metrics_fold['Algorithm'] == 'consensus'],
                   palette='gray', alpha=0.2, width=0.3, dodge=False, fill=True, linewidth=0, inner=None, legend=False, ax=ax)
    sns.boxplot(x='Features', y=metric, hue='Algorithm',
                data=df_metrics_fold[df_metrics_fold['Algorithm'] == 'consensus'],
                palette='gray', width=0.3, gap=0.2, dodge=False, fill=False, linewidth=1.5, fliersize=1, whis=[2.5, 97.5], ax=ax)
    ax.axhline(y=bench, linestyle='--', color='gray', linewidth=1)
    sns.move_legend(fig, "upper left", bbox_to_anchor=(1, 1.01))
    plt.xlabel("")
    plt.ylabel(metric)
    plt.ylim([llim, ulim])
    plt.tight_layout()
    plt.rcParams.update({'font.size': fontsize})

    return fig


def plot_rmse_nn_architecture(data, x='layers', xlabel='Nr of hidden layers', ax=None, fontsize=10, legend=False):
    """ Creates a combined violin and boxplot to compare the distribution of RMSE across crossvalidation folds for 
    different neural network architectures (number or size of hidden layers)

    Inputs
    ----------
    data : pandas dataframe, mandatory
        Combined structured config files from visualize_NN-architecture.py containing error metrics for different architectures in every crossvalidation fold
    x: str, mandatory
        The architecture aspect by which to group the data, e.g. 'layers' or 'nodes'
    xlabel : str, mandatory
        The label of the x-axis

    Outputs
    ----------
    ax: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
       
    palette = {'train': 'teal', 'test': 'mediumblue'}

    # plt.axes(ax)

    sns.violinplot(x=x, y='RMSE', hue='set', data=data, #order=order, hue_order=hue_order, 
                       palette=palette, alpha=0.5, width=0.9, dodge=True, fill=True, linewidth=0, inner=None, legend=legend, ax=ax)
    
    sns.boxplot(x=x, y='RMSE', hue='set', data=data, #order=order, hue_order=hue_order, 
                palette=palette, width=0.9, gap=0.4, dodge=True, fill=False, linewidth=1.5, fliersize=1, whis=[2.5, 97.5], legend=False, ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('RMSE [log10(mg/kg-d)]')

    if legend:
        sns.move_legend(ax, "center left", bbox_to_anchor=(1.02, 0.5), title="data set")

    ax.grid(color='lightgrey', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    plt.rcParams.update({'font.size': fontsize})

    return ax


def plot_chemical_space_classes(df):
    """ Creates the two-dimensional t-SNE embedding of the marketed chemicals colored by ClassyFire Superclasses 
    (Djoumbou et al., 2016) following von Borries et al. (2023).

    Inputs
    ----------
    data : pandas dataframe, mandatory
        Structured dataframe from visualize_chem-space.py containing t-SNE coordinates and ClassyFire Classification 
        for the large set of >130,000 marketed chemicals

    Outputs
    ----------
    ax: matplotlib.axes.Axes
        The matplotlib axes containing the plot.
    """
    
    top15 = df.groupby('Superclass').count()['TSNE1'].sort_values(ascending=False).index[:15]
    df['Superclass_top15'] = df['Superclass'].where(df['Superclass'].isin(top15), 'Other')

    # Plot figure
    c = 'Superclass_top15'

    palette = ['darkslategrey', 'teal', 'aquamarine', 'darkred', 'orangered', 'mediumpurple', 'darkorchid',
            'mediumblue', 'royalblue', 'skyblue', 'darkgoldenrod', 'darkorange', 'gold',  'lightpink', 'hotpink',
            'lightgrey']

    order = top15.sort_values().to_list() + ['Other']

    fig, ax = plt.subplots(figsize=(18, 10))
    plt.subplots_adjust(left=0.1, right=0.65)
    sns.scatterplot(df, x='TSNE1', y='TSNE2', hue=c, hue_order=order, palette=palette, edgecolor=None, alpha=1, s=2, legend=None)
    
    ax.axis('off')

    # custom legend
    occurrence = df[c].value_counts(normalize=True)
    ax1 = inset_axes(ax,
                    width=1,  # inch
                    height=9,  # inch
                    bbox_transform=ax.transAxes,  # relative axes coordinates
                    bbox_to_anchor=(1.25, 1.1),  # relative axes coordinates
                    loc=2)  # loc=lower left corner

    width, height, pad = 0.5, 20, 0.25
    y = 0
    ax1.axis('off')

    for n, s in enumerate(order[::-1]):
        h = max(height * occurrence[s], 0.1)
        ax1.add_patch(patches.Rectangle((0, y), width, h, facecolor=palette[::-1][n]))
        ax1.text(-0.5, y + h / 2 - 0.1, "{:.1%}".format(occurrence[s]),
                fontdict={'size': 10})
        ax1.text(width + 0.1, y + h / 2 - 0.1, s,
                fontdict={'size': 10})
        y = y + h + pad

    ax1.text(-0.5, y + h / 2 + 0.1, "$\\bf{Total\ number\ of\ chemicals}$: " + str(df.shape[0]),
                fontdict={'size': 10})

    plt.xlim([0, 1])
    plt.ylim([0, y])

    return fig