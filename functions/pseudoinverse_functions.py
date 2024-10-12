'''
General and misc. functions that help with data management
'''
from scipy import stats
import statsmodels.api as sm
import statsmodels.stats as sms
from sklearn.metrics import mean_absolute_error

import numpy as np
import pandas as pd

 
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

    # sort the values
    df = data.copy()
    df['rank_sort'] = rank_sort
    df = df.sort_values(by='rank_sort')
    df = df.drop(columns='rank_sort')

    # get the correlation for every combination of traits
    trait_corr = df.T.corr(method='spearman').abs()

    # mask the diagonals and upper triangle
    mask = np.tril(np.ones(trait_corr.shape), k=-1).astype(bool)
    masked_corr = trait_corr.where(mask, 0)

    # for the traits we want to keep
    # we do two loops so they cant interact with eachother
    for val in keep_val: masked_corr[val] = trait_corr[val] # make their columns unmasked so they participate in filtering
    for val in keep_val: masked_corr.loc[val] = 0 # make their rows 0 so they can't be removed

    # make all values > filter = nan, then drop their corresponding rows
    dropped_corr = masked_corr.copy()
    dropped_corr[dropped_corr > filter] = np.nan
    dropped_corr = dropped_corr.dropna(axis=0)

    data = data.loc[data.index.isin(dropped_corr.index)]

    return data, masked_corr

def pinv_iteration(trait_data, meth_data, pred_trait=True):
    '''
    utilizes leave 1 out cross validation, gives accuracy of calculation via pseudoinversion by trait

        param trait_data: trait-associated data, m = traits, n = animal/individual, df
        param meth_data: methylation score data, m = probe ID, n = animal/individual, df
        pred_trait: determines whether the traits (True) or probes (False) should be the dependent variable

        return: 3 dictionaries- y_train, y_test, index, as dictionaries by index name (site or trait)

    '''

    # to match reference paper (https://github.com/giuliaprotti/Methylation_BuccalCells/tree/main)
    trait_data_T = trait_data.T # animals x traits
    meth_data_T = meth_data.T # animals x factors

    # add a constant = 1 to the trait matrix
    trait_data_T['bias_term'] = 1

    # convert to numPy arrays
    trait_vals = trait_data_T.values
    meth_vals = meth_data_T.values

    if pred_trait: # if predicting trait
        X = meth_vals.copy() # the dataset used for making predictions (the independent variable)
        y = trait_vals.copy() # the dataset for which predictions are made (the dependent variable)
        y_names = list(trait_data.index) # not the transposed variant so we get the actual trait names
        
    else: # if predicting methylation
        X = trait_vals.copy()
        y = meth_vals.copy()
        y_names = list(meth_data.index)

    # dictionaries for the predictions
    all_pred = {var: [] for var in y_names}
    all_actual = {var: [] for var in y_names}
        
    for i in range(y.shape[0]): # for every animal

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

        # add values to their respective dictionaries
        # note that the bias term is excluded
        m = 0
        while m < len(y_names): # the first value of the predictions corresponds to the first site or trait, etc. (can be tested by keeping as dfs)
            all_pred[y_names[m]].append(pred[m])
            all_actual[y_names[m]].append(y_test[m])
            m+=1

    return all_pred, all_actual

def pinv_dropmin(trait_data, meth_data, trait_thresh, 
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

        pred, actual = pinv_iteration(trait_data, meth_data)
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

    # get the pvalues and coefficients for each trait/site combination
    trait_pvals, trait_vals = meth_calc(trait_data, meth_data)
    
    return pred, actual, trait_vals, trait_pvals

def filter_meth(trait_data, meth_data, thresh=0.5):
    '''
    filters methylation data, removing those probes which do not vary significantly between individuals

        param trait_data: trait-associated data, m = traits, n = animal/individual, df
        param meth_data: methylation score data, m = probe ID, n = animal/individual, df
        param thresh: threshold for dropping probes, if mean absolute error (actual vs predicted) / std, drop

        return: filtered methylation data, df
    '''

    pred, actual = pinv_iteration(trait_data, meth_data, pred_trait=False)

    to_remove = []
    for key in pred.keys():
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
    
    for i, probe in enumerate(meth_vals): # get p values for all probe-trait combinations (i.e. the "multiple" in multiple linear regression)
        model = sm.OLS(probe, X).fit() #  the "linear" in multiple linear regression
        pvals[i] = model.pvalues[1:] # get the p values from the OLS, excluding the intercept
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
        print(pvals_by_trait[n].shape)
        adj_pvals = sms.multitest.fdrcorrection(pvals_by_trait[n], alpha=0.01) # adjust the pvals by trait
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
    '''
    Counts the number of non-NaN rows for two specified columns, with overlapping non-NaN rows counted once.

    param df: The DataFrame containing the data.
    param col1: The name of the first column.
    param col2: The name of the second column.

    return: none
    '''
    # Ensure the columns exist in the DataFrame
    if col1 in df.columns and col2 in df.columns:
        non_nan_count = df[[col1, col2]].notna().any(axis=1).sum()
        print(f'{col1} and {col2}: {non_nan_count}')
    else:
        print(f"One or both columns '{col1}' and '{col2}' are not in the DataFrame")

    return