'''
General and misc. functions that help with data management
'''
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats as sms
from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import string

def quality_filter(data, filter, keep_val=['Rank', 'CD1 or C57BL6J?', 'C57BL6J or Sv129Ev?']):
    '''
    removes highly similar data, determined by spearman coefficient

        param data: data to be assessed, by row, df
        param filter: spearman coefficient threshhold, float
        param keep_val: which traits to forcefully keep

        return: filtered data, df
    '''

    # create a dictionary indicating sort order
    timepoint_rank_map = {
        'M16': 0,
        'M14': 1,
        'M12': 2,
        'M10': 3,
        'M8': 4,
        'M6': 5,
        'w20': 6,
        'w18': 7,
        'w16': 8,
        'w15': 9,
        'w14': 10,
        'w13': 11,
        'w12': 12
    }

    # sort data so that later entries are prioritized, since methylation data was collected at M17
    rank_sort = []
    for column in data.index:
        timepoint = column.split('_')[0]
        # use the mapping to get the rank, defaulting to 13 for unknown timepoints
        rank = timepoint_rank_map.get(timepoint, 13)
        rank_sort.append(rank)

    df = data.copy()
    df['rank_sort'] = rank_sort
    df = df.sort_values(by='rank_sort')
    df = df.drop(columns='rank_sort')

    index = df.index
    df = df.values # decreases runtime

    # filters highly colinear traits
    n = 0
    while n < len(df):
        m = 0
        #print(f'\ninitializing: {index[n]}\n')
        while m < len(df):
            if m == n: # if both selectors are on the same trait
                m+=1 # skip equivalent
                if m >= len(df): # if doing so leads to surpassing the df, break the loop
                    break
                else:
                    pass
            corr = stats.spearmanr(df[n], df[m]) # find similarity between two traits
            #print(f'{index[m]} corr: {abs(corr[0])}')
            if abs(corr[0]) > filter: # if colinearity is high
                if index[m] not in keep_val: # if not dealing with a special case
                    #print(f'\nremoving: {index[m]}\n')
                    df = np.delete(df, m, 0) # remove where m is selected from df
                    index = np.delete(index, m, 0) # shifts m=4, for instance to being equivalent to what was previously m=5
                    if n >= len(df)-1: # if last iteration, n needs to be shifted aswell to account for deletion
                        n-=1   
                    m-=1 # because m will be increased by 1 at the end of the loop
            m+=1
        n+=1

    df = pd.DataFrame(df) 
    df = df.set_index(index)

    return df

def corr_scatter(pred, actual):
    '''
    outputs scatterplots comparing predictions vs actual values

        param pred: data indicating predicted trait values, dict
        param actual: data indicating actual trait values, dict

        return: none
    '''

    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(20, 15)) # not generalizeable, adjust as needed

    keys = list(pred.keys())
    for i, ax in enumerate(axs.flat):
        try: # in case you have an odd number of inputs, i.e. 5
            x = pred[keys[i]]
            y = actual[keys[i]]

            sns.scatterplot(data=None, x=x, y=y, alpha = 0.5, color='blue', ax=ax)
            ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='rebeccapurple')
            
            info = stats.spearmanr(x, y)
            corr_coef = '{0:.3f}'.format(abs(info[0]))

            ax.set_title(f'|{keys[i]}| = {corr_coef}', size=15, pad=10)
            ax.set_xlabel('model pred', size=15)
            ax.set_ylabel('actual', size=15)
            ax.text(-0.1, 1.1, string.ascii_uppercase[i], transform=ax.transAxes, size=20, weight='bold')
            
        except (KeyError, IndexError) as e:
            fig.delaxes(ax)

    plt.subplots_adjust(left=0.125, bottom=0.1, right=1.1,
                        top=1.0, wspace=0.2, hspace=0.4)
        
    return

def probe_heatmap(data):
    '''
    generates a heatmap using methylation p values

        param data: data from which to generate heatmap, df

        returns: none
    '''

    fig, axs = plt.subplots(figsize=(8,10))

    sns.heatmap(data, norm=LogNorm(vmin = 0.001, vmax = 0.5),
                cmap='YlGnBu_r', cbar_kws={'extend': 'max'}, ax=axs)
    
    axs.set_ylabel(f'{data.shape[0]} CpG sites (p value)', fontsize=20)
    axs.set_xlabel('trait', fontsize=20)

    axs.set_yticks([])
    axs.tick_params('x', rotation=90, size=15)

    plt.tight_layout()

    return

def trait_cluster(data, trait, n_probes):
    '''
    generates clustermap of p values

        param data: p value data, m = probes, n = traits, df or dict
        param trait: specific trait for which top inputs are selected, str
        param n_probes: number of probes to be selected for a given trait, from lowest to highest pval, int

        return: none
    '''

    # convert dict to df
    if isinstance(data, dict): data = pd.DataFrame(data)

    clustermap_df = data.sort_values(by=trait)
    clustermap_df = clustermap_df.iloc[:n_probes]
    clustermap_df = clustermap_df.fillna(1)

    cluterm = sns.clustermap(clustermap_df, cmap='YlGnBu_r', vmin=0.0001, vmax=0.01, mask=(clustermap_df==1))
    cluterm.ax_heatmap.set_title(f'top {n_probes} {trait}', fontsize=15)

    cluterm.ax_heatmap.set_ylabel(f'{n_probes} CpG sites (p value)', fontsize=15)
    cluterm.ax_heatmap.set_xlabel('trait', fontsize=15)
    cluterm.ax_heatmap.tick_params(axis='x', labelsize=12, rotation=90)
    
    cluterm.ax_heatmap.set_facecolor('palegreen')

    plt.tight_layout()
    plt.show()

    return

def pinv_iteration(trait_data, meth_data, pred_trait=True):
    '''
    utilizes leave 1 out cross validation, gives accuracy of calculation via pseudoinversion by trait

        param trait_data: trait-associated data, m = traits, n = animal/individual, df
        param meth_data: methylation score data, m = probe ID, n = animal/individual, df
        pred_trait: determines whether the traits (True) or probes (False) should be the dependent variable

        return: 3 dictionaries- y_train, y_test, index, as dictionaries by index name (site or trait)

    '''

    trait_vals = trait_data.values
    meth_vals = meth_data.values

    # to match reference paper (https://github.com/giuliaprotti/Methylation_BuccalCells/tree/main)
    trait_vals = trait_vals.T
    meth_vals = meth_vals.T

    if pred_trait: # if predicting trait
        X = meth_vals.copy() # the dataset used for making predictions (the independent variable)
        y = trait_vals.copy() # the dataset for which predictions are made (the dependent variable)
        y_names = list(trait_data.index)
    else: # if predicting methylation
        X = trait_vals.copy()
        y = meth_vals.copy()
        y_names = list(meth_data.index)

    all_pred = {var: [] for var in y_names}
    all_actual = {var: [] for var in y_names}

    # add a constant term = 1, so we dont force the model to have 0 for meth when trait = 0
    X = sm.add_constant(X, prepend=False) # add it too the end
        
    for i in range(len(y)): # so that the constant isn't reached

        # define mask to distinguish train and test set
        mask = np.arange(len(y)) != i

        # leave out one
        X_train = X[mask]
        X_test = X[i]
        y_train = y[mask]
        y_test = y[i]

        # generate predictions 
        if pred_trait:
            # this formula is only valid when there are less traits than observations for y_train
            site_coef = np.matmul(np.linalg.pinv(y_train), X_train) # Coefficient = Pinv(Trait_Train) * Meth_Train
            pred = np.matmul(X_test, np.linalg.pinv(site_coef)) # Trait_Pred = Meth_Test * Pinv(Site Coef)
        else:
            site_coef = np.matmul(np.linalg.pinv(X_train), y_train) # Coefficient = Pinv(Trait_Train) * Meth_Train
            pred = np.matmul(X_test, site_coef) # Meth_Pred = Trait_Test * Site Coef

        # add values to the dictionary
        m = 0
        while m < len(y_names):
            all_pred[y_names[m]].append(pred[m])
            all_actual[y_names[m]].append(y_test[m])
            m+=1

    return all_pred, all_actual, y_names

def pinv_dropmin(trait_data, meth_data, trait_thresh, find_meth=False, plot_results=True, 
                 probe_thresh=0, to_keep = ['Rank']):
    '''
    identifies those traits highly predictable using methylation data,
    and uses this information according to parameter settings

        param trait_data: trait-associated data, m = traits, n = animal/individual, df
        param meth_data: methylation score data, m = probe ID, n = animal/individual, df
        param trait_thresh: threshold for dropping traits, if accuracy < thresh, drop trait and loop function, float
        param find_meth: if True, intiaites additional multivariate regression for each remaining trait/probe combination, bool
        param plot_results: if True, plots results of data analysis, in accordance with other parameters, bool
        param probe_thresh: threshold of mean difference for dropping methylation sites, if val > param, drop
        param to_keep: which traits to keep, as a list

        return: 3 dictionaries- if find_meth = False, keys = traits, vals = model predictions, actual, index,
                            else, keys = probes, vals = pvals+coefs, pvals, coefs 
    '''

    if probe_thresh != 0: # decrease number of methylation probes
        meth_data = filter_meth(trait_data, meth_data, probe_thresh)

    any_dropped = True # to initiate the loop
    while any_dropped:

        pred, actual, index = pinv_iteration(trait_data, meth_data)
        any_dropped = False # none have been dropped yet

        to_remove = []
        for key in pred.keys():
            if key in to_keep: # skip the traits which are being forcefully maintained
                continue
            corr = stats.spearmanr(pred[key], actual[key]) # get the prediction accuracy
            if abs(corr[0]) < trait_thresh: # if the absolute value of the correlation coefficient is under the threshhold
                to_remove.append(key) # prepare to drop the poorly predicted traits
                any_dropped = True # tells us that some have been dropped, so we should continue running iterations
        trait_data = trait_data.drop(index=to_remove) # drop the poorly predicted traits

    if find_meth:
        trait_pvals, trait_vals = meth_calc(trait_data, meth_data)
        if plot_results: 
            probe_heatmap(pd.DataFrame.from_dict(trait_pvals))
        return trait_vals, trait_pvals
    
    elif plot_results: 
        corr_scatter(pred, actual)

    return pred, actual, index

def filter_meth(trait_data, meth_data, thresh=0.5):
    '''
    filters methylation data, removing those probes which do not vary significantly between individuals

        param trait_data: trait-associated data, m = traits, n = animal/individual, df
        param meth_data: methylation score data, m = probe ID, n = animal/individual, df
        param thresh: threshold for dropping probes, if mean absolute error (actual vs predicted) / std, drop

        return: filtered methylation data, df
    '''

    pred, actual, index = pinv_iteration(trait_data, meth_data, pred_trait=False)

    to_remove = []
    for key in index:
        temp = (mean_absolute_error(actual[key], pred[key]) / np.std(actual[key])) # mean abs error / std
        if temp >= thresh: # i.e keep those <thresh
            to_remove.append(key)
        else:
            pass

    # remove the probes with poor predication accuracy
    meth_data = meth_data.drop(to_remove)
    return meth_data

def meth_calc(trait_data, meth_data):
    '''
    runs MLR, X (dependent) = probes, y (independent) = traits
    gets AdjP for each trait/probe combination
    
        param trait_data: trait-associated data, m = traits, n = animal/individual, df
        param meth_data: methylation score data, m = probe ID, n = animal/individual, df

        return: 2 dictionaries- keys = traits, vals = model pvals, model coefficients
    '''
    meth_index = list(meth_data.index)
    trait_names = list(trait_data.index)

    trait_vals = trait_data.values
    trait_vals = trait_vals.T
    meth_vals = meth_data.values

    # create the empty arrays to store the pval and coefficient information
    pvals = np.zeros((meth_vals.shape[0], trait_vals.shape[1]), dtype='float32')
    coef = pvals.copy()

    # add a constant term to trait_vals so that the model is fit through an origin of 1
    X = sm.add_constant(trait_vals, prepend=True) # adds to the first column
    
    for i, probe in enumerate(meth_vals): # get p values for all probe-trait combinations (i.e. the "multiple" in mulitple multivariate regression)
        model = sm.OLS(probe, X).fit() #  the "univariate" in univariate/linear multivariate regression
        pvals[i] = model.pvalues[1:]
        coef[i] =  model.params[1:]

    # so that we can iterate by trait
    pvals_by_trait = pvals.T
    coef_by_trait = coef.T

    # create dictionary for trait-wise adjusted values
    trait_pvals = {}
    trait_coefs = {}

    # adjust the p values and add them + the coefficients to respective dictionary
    n = 0
    while n < len(trait_names):

        adj_pvals = sms.multitest.fdrcorrection(pvals_by_trait[n], alpha=0.05) # adjust the pvals by trait
        trait_pvals[trait_names[n]] = adj_pvals[1] # adj_pvals gives multiple arrays, we want the actual values
        trait_coefs[trait_names[n]] = coef_by_trait[n]

        n+=1

    # make a dataframe representing all of the trait/probe combinations and their adjusted values
    trait_all_vals = pd.DataFrame()
    for key in trait_pvals.keys():
        trait_all_vals[f'{key}_pval'] = trait_pvals[key]
        trait_all_vals[f'{key}_coef'] = trait_coefs[key]

    # make the index the probe names
    trait_all_vals  = trait_all_vals.set_index(pd.Index(meth_index))

    return trait_pvals, trait_all_vals

def count_cumulative_probes(df, col1, col2):
    """
    Counts the number of non-NaN rows for two specified columns, with overlapping non-NaN rows counted once.

    param df: The DataFrame containing the data.
    param col1: The name of the first column.
    param col2: The name of the second column.

    return: none
    """
    # Ensure the columns exist in the DataFrame
    if col1 in df.columns and col2 in df.columns:
        non_nan_count = df[[col1, col2]].notna().any(axis=1).sum()
        print(f'{col1} and {col2}: {non_nan_count}')
    else:
        print(f"One or both columns '{col1}' and '{col2}' are not in the DataFrame")

    return
