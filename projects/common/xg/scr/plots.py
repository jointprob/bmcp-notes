from itertools import combinations

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import scipy.stats as stats

def quantiles(df_orig, cat_col, var_col, q_method='inverted_cdf'):
    
    cat_dfs = []
    for cat, cat_df in df_orig.groupby(by=cat_col):
        cat_df = cat_df.sort_values(by=var_col)
        i = np.arange(0, len(cat_df), 1) + 1
        cat_df['f-value'] = (i - 0.5) / len(cat_df)
        # cat_df[var_col] = np.quantile(cat_df[var_col], cat_df['f-value'].values, method=q_method)
        cat_dfs.append(cat_df)
    f_df = pd.concat(cat_dfs, axis=0).sort_values(by=[cat_col, var_col]).reset_index(drop=True)

    return f_df


def theoretical_qq(df, cat_col, var_col, dist='norm'):
    cat_type=CategoricalDtype(categories=df[cat_col].cat.categories, ordered=True)
    
    qdfs = []
    for cat, cat_df in df.groupby(by=cat_col):
        xys, line_def = stats.probplot(cat_df[var_col].values, dist=dist)
        qdf = pd.DataFrame({cat_col: cat, 'theory_quantiles': xys[0], var_col: xys[1],
                            'slope': line_def[0], 'intercept': line_def[1]})
        qdfs.append(qdf)
        
    df = pd.concat(qdfs, axis=0).reset_index(drop=True)
    df[cat_col] = df[cat_col].astype(cat_type)
    return df 


def qq_quantiles(df, cat_col, var_col, cols):
    x = df[df[cat_col]==cols[0]][var_col].rename(str(cols[0]) + ' ' + var_col)
    y = df[df[cat_col]==cols[1]][var_col].rename(str(cols[1]) + ' ' + var_col)
    # deal with same length arrays
    if len(x)==len(y):
        min_len_arr = x
        max_len_arr = y
    else: # deal with different length arrays
        min_len_arr = [x, y][np.argmin([len(x), len(y)])]
        max_len_arr = [x, y][np.argmax([len(x), len(y)])]
    x_name = min_len_arr.name
    y_name = max_len_arr.name
    min_len = len(min_len_arr)
    
    min_len_arr = np.sort(min_len_arr)
    i = np.arange(0, min_len, 1) + 1
    f_i = (i - 0.5) / min_len
    
    # max_len_arr_q = np.quantile(max_len_arr, f_i, method='linear')
    max_len_arr_q = np.quantile(max_len_arr, f_i, interpolation='linear')

    return pd.DataFrame({x_name: min_len_arr, y_name: max_len_arr_q})


def mean_diffs(qq):
    # Modify to deal with categories
    qq['mean'] = qq.mean(axis=1)
    qq['diff'] = qq.apply(lambda x: x[1] - x[0], axis=1)
    return qq


def pairwise_qq(df, cat_col, var_col):
    
    cat_type=CategoricalDtype(categories=df[cat_col].cat.categories, ordered=True)
    # Use unique to get actual categories in the df data, in case some are absent in the df
    pairs = combinations(df[cat_col].unique(), 2)
    
    dfs = []
    for pair in pairs:
        qq = qq_quantiles(df, cat_col, var_col, pair)
        qq['row'] = pair[0]
        qq['col'] = pair[1]
        # Deal with the way that qq_quantiles function names the columns
        qq = qq.rename(columns={col: col.replace(" " + var_col, "") for col in qq.columns})
        # Make a dictionary - use it for col renaming
        lookup_d = {qq['row'].unique()[0]: 'row', qq['col'].unique()[0]: 'col'}
        qq = qq.rename(columns={col: lookup_d[col] + '_vals' for col in qq.columns if col not in ['row', 'col']})
        dfs.append(qq)
        
    df = pd.concat(dfs, axis=0).reset_index(drop=True)
    df['col'] = df['col'].astype(cat_type)
    df['row'] = df['row'].astype(cat_type)
    return df
