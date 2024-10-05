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

    # sort data so that later entries are prioritized, since methylation data was collected at M17
    
    rank_sort = []
    for column in data.index:
        if 'M16_poststress_' in column: rank_sort.append(0)
        elif 'M14_poststress_' in column: rank_sort.append(1)
        elif 'M12_poststress_' in column: rank_sort.append(2)
        elif 'M10_poststress_' in column: rank_sort.append(3)
        elif 'M8_poststress_' in column: rank_sort.append(4)
        elif 'M6_poststress_' in column: rank_sort.append(5)
        elif 'w20_poststress_' in column: rank_sort.append(6)
        elif 'w18_poststress_' in column: rank_sort.append(7)
        elif 'w16_stress_' in column: rank_sort.append(8)
        elif 'w15_stress_' in column: rank_sort.append(9)
        elif 'w14_stress_' in column: rank_sort.append(10)
        elif 'w13_stress_' in column: rank_sort.append(11)
        elif 'w12_baseline' in column: rank_sort.append(12)
        else: rank_sort.append(13)

    df = data.copy()
    df['rank_sort'] = rank_sort
    df = df.sort_values(by='rank_sort')
    df = df.drop(columns='rank_sort')

    index = df.index
    df = df.values # decreases runtime

    n = 0
    while n < len(df):
        m = 0
        print(f'\ninitializing: {index[n]}\n')
        while m < len(df):
            if m == n: # if both selectors are on the same trait
                m+=1 # skip equivalent
                if m >= len(df): # if doing so leads to surpassing the df, break the loop
                    break
                else:
                    pass
            corr = stats.spearmanr(df[n], df[m]) # find similarity between two traits
            print(f'{index[m]} corr: {abs(corr[0])}')
            if abs(corr[0]) > filter: # if colinearity is high
                if index[m] not in keep_val: # if not dealing with a special case
                    print(f'\nremoving: {index[m]}\n')
                    df = np.delete(df, m, 0) # remove where m is selected from df
                    index = np.delete(index, m, 0) # shifts m=4, for instance to being equivalent to what was previously m=5
                    if n >= len(df)-1: # if last iteration, n needs to be shifted aswell to account for deletion
                        n-=1   
                    m-=1 # because m will be increased by 1 at the end of the loop
            m+=1
        n+=1

    '''
    note that the last trait won't be initialized, here that is CD1 or C57BL6J?,
    but that isn't an issue because every trait has already been compared to it
    '''

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
        try:

            x = pred[keys[i]]
            y = actual[keys[i]]

            sns.scatterplot(data=None, x=x, y=y, alpha = 0.5, color='blue', ax=ax)

            ax.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='rebeccapurple')
            
            info = stats.spearmanr(x, y)
            corr_coef = '{0:.3f}'.format(info[0])
            ax.set_title(f'|{keys[i]}| = {corr_coef}', size=15, pad=10)

            ax.set_xlabel('model pred', size=15)
            ax.set_ylabel('actual', size=15)

            ax.text(-0.1, 1.1, string.ascii_uppercase[i], transform=ax.transAxes, 
                size=20, weight='bold')
            
        except:
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

        param data: p value data, m = probes, n = traits, df
        param trait: specific trait for which top inputs are selected, str
        param n_probes: number of probes to be selected for a given trait, from lowest to highest pval, int

        return: none
    '''

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
        pred_trait: determines whether the traits (True) or probes (False) should be the variable solved for

        return: 3 dictionaries- keys = traits or probes (depending on pred_trait), vals = model predictions, actual, index

    '''

    trait_names = list(trait_data.index)
    probe_names = list(meth_data.index)

    trait_data = trait_data.values
    meth_data = meth_data.values

    all_pred = {}
    all_actual = {}

    # to match reference paper (https://github.com/giuliaprotti/Methylation_BuccalCells/tree/main)
    trait_data = trait_data.T
    meth_data = meth_data.T

    if pred_trait:

        for trait in trait_names:
            all_pred[trait] = []
            all_actual[trait] = []
        
        for i in range(len(trait_data)):

            trait_train = np.delete(trait_data, i, axis=0)
            trait_test = trait_data[i]

            meth_train = np.delete(meth_data, i, axis=0)
            meth_test = meth_data[i]

            # add column of constants for pinv, so we dont force the model to have 0 for meth when trait = 0
            trait_train = np.append(trait_train,np.ones([len(trait_train),1]),1)

            # this formula is only valid when there are less traits in trait_train than observations

            site_coef = np.matmul(np.linalg.pinv(trait_train), meth_train) # Coefficient = Pinv(Trait_Train) * Meth_Train
            trait_pred = np.matmul(meth_test, np.linalg.pinv(site_coef)) # Trait_Pred = Meth_Test * Pinv(Site Coef)

            m = 0
            while m < len(trait_names):
                all_pred[trait_names[m]].append(trait_pred[m])
                all_actual[trait_names[m]].append(trait_test[m])
                m+=1

        return all_pred, all_actual, trait_names
    
    else: # pred meth

        for probe in probe_names:
            all_pred[probe] = []
            all_actual[probe] = []

        for i in range(len(meth_data)):

            trait_train = np.delete(trait_data, i, axis=0)
            trait_test = trait_data[i]

            meth_train = np.delete(meth_data, i, axis=0)
            meth_test = meth_data[i]

            # add column of constants for pinv, so we dont force the model to have 0 for trait when meth = 0
            meth_train = np.append(meth_train,np.ones([len(meth_train),1]),1)

            site_coef = np.matmul(np.linalg.pinv(trait_train), meth_train) # Coefficient = Pinv(Trait_Train) * Meth_Train
            meth_pred = np.matmul(trait_test, site_coef) # Meth_Pred = Trait_Test * Site Coef

            m = 0
            while m < len(probe_names):
                all_pred[probe_names[m]].append(meth_pred[m])
                all_actual[probe_names[m]].append(meth_test[m])
                m+=1

        return all_pred, all_actual, probe_names

def pinv_dropmin(trait_data, meth_data, thresh, find_meth=False, 
                 plot_results=True, to_filter_meth=False, meth_filter_thresh=0):
    '''
    identifies those traits highly predictable using methylation data,
    and uses this information according to parameter settings

        param trait_data: trait-associated data, m = traits, n = animal/individual, df
        param meth_data: methylation score data, m = probe ID, n = animal/individual, df
        param thresh: threshold for dropping traits, if accuracy < thresh, drop trait and loop function, float
        param find_meth: if True, intiaites additional multivariate regression for each remaining trait/probe combination, bool
        param polt_results: if True, plots results of data analysis, in accordance with other parameters, bool
        param to_filter_meth: optimizes methylation sites before usage in data analysis, bool
        param meth_filter_thresh: threshold of mean difference for dropping methylation sites, if val > param, drop

        return: 3 dictionaries- if find_meth = False, keys = traits, vals = model predictions, actual, index,
                            else, keys = probes, vals = pvals+coefs, pvals, coefs 
    '''

    if to_filter_meth:
        meth_data = filter_meth(trait_data, meth_data, meth_filter_thresh)
    else:
        pass

    loop_exit = False
    while not loop_exit:

        to_drop = []
        pred, actual, index = pinv_iteration(trait_data, meth_data)

        corr_vals = []

        for key in pred.keys(): # find the spearman for each trait
            corr = stats.spearmanr(pred[key], actual[key])
            corr_vals.append(corr[0])

        n = 0 # prepare to drop low spearman values
        while n < len(corr_vals):
            if abs(corr_vals[n]) < thresh:
                to_drop.append(index[n])
            else:
                pass
            n+=1

        try: # forcefully keep Rank
            to_drop.remove('Rank')
        except ValueError:
            pass

        for trait in to_drop: # drop others
            trait_data = trait_data.drop(trait, axis=0)

        if to_drop == []: # if nothing left to drop, complete loop
            loop_exit = True
        else:
            pass

    if find_meth: # find probes for remaining traits
        trait_pvals, trait_coefs = meth_calc(trait_data, meth_data)

        trait_pvals = pd.DataFrame.from_dict(trait_pvals)
        trait_coefs = pd.DataFrame.from_dict(trait_coefs)

        trait_vals = pd.DataFrame()
        for column in trait_pvals.columns:
            trait_vals[f'{column}_pval'] = trait_pvals[column]
            trait_vals[f'{column}_coef'] = trait_coefs[column]

        if plot_results: probe_heatmap(trait_pvals)

        return trait_vals, trait_pvals, trait_coefs
    
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

    meth_data = meth_data.drop(to_remove)
    
    return meth_data

def meth_calc(trait_data, meth_data):
    '''
    runs multivariate regression with the dependent variable being the probe data, 
    and the independent variables being the traits, producing p values for each trait/probe combination  
    
        param trait_data: trait-associated data, m = traits, n = animal/individual, df
        param meth_data: methylation score data, m = probe ID, n = animal/individual, df

        return: 2 dictionaries- keys = traits, vals = model pvals, model coefficients
    '''
    meth_index = list(meth_data.index)

    trait_names = trait_data.index
    trait_data = trait_data.T

    trait_data = trait_data.values
    meth_data = meth_data.values

    pvals = [0]*trait_data[0].shape[0]
    pvals = np.array(pvals, dtype='float32')

    coef = [0]*trait_data[0].shape[0]
    coef = np.array(pvals, dtype='float32')

    for probe in meth_data: # get p values for all probe-trait combinations
        model = sm.OLS(probe, trait_data).fit()
        pvals = np.vstack((pvals, model.pvalues))
        coef = np.vstack((coef, model.params))

    pvals = np.delete(pvals, 0, 0)
    pvals = pvals.T

    coef = np.delete(coef, 0, 0)
    coef = coef.T 

    trait_pvals = {}
    trait_coefs = {}
    n = 0
    while n < len(trait_names):
        adj_pvals = sms.multitest.fdrcorrection(pvals[n], alpha=0.05)

        trait_pvals[trait_names[n]] = adj_pvals[1]
        trait_coefs[trait_names[n]] = coef[n]

        n+=1

    trait_pvals = pd.DataFrame.from_dict(trait_pvals)
    trait_pvals['index'] = meth_index
    trait_pvals = trait_pvals.set_index('index')

    trait_coefs = pd.DataFrame.from_dict(trait_coefs)
    trait_coefs['index'] = meth_index
    trait_coefs = trait_coefs.set_index('index')

    return trait_pvals, trait_coefs