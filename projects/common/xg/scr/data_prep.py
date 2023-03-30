import json

import pandas as pd
import numpy as np

# https://stackoverflow.com/questions/52795561/flattening-nested-json-in-pandas-data-frame
def flatten_json(nested_json, exclude=['']):
    """Flatten json object with nested keys into a single level.
        Args:
            nested_json: A nested json object.
            exclude: Keys to exclude from output.
        Returns:
            The flattened json object if successful, None otherwise.
    """
    out = {}

    def flatten(x, name='', exclude=exclude):
        if type(x) is dict:
            for a in x:
                if a not in exclude: flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(nested_json)
    return out

def load_events(filepath):
    with open(str(filepath)) as f:
        events_dict = {'events':json.load(f)}
    df = pd.DataFrame([flatten_json(x) for x in events_dict['events']])
    df['subEventName'] = df['subEventName'].str.replace(' ', '', regex=True)
    return df

def filter_on_subEvent(df_orig, subEventName):
    df = df_orig.copy()
    df = df[df['subEventName']==subEventName].reset_index(drop=True)
    return df
    
def id_col_on_tag(df_orig, tag_num, new_col_name):
    # 6 tags per row
    df = df_orig.copy()
    crit1 = df['tags_0_id']==tag_num
    crit2 = df['tags_1_id']==tag_num
    crit3 = df['tags_2_id']==tag_num
    crit4 = df['tags_3_id']==tag_num
    crit5 = df['tags_4_id']==tag_num
    crit6 = df['tags_5_id']==tag_num
    df[new_col_name] = (df[crit1 | crit2 | crit3 | crit4 | crit5]).any(1)
    df[new_col_name] = df[new_col_name].fillna(False).astype(int)
    return df


def get_grid_coords(df_orig):
    df = df_orig.copy()
    return df

def make_geometry(df_orig):
    df = df_orig.copy()
    df['X'] = 100 - df['positions_0_x']
    df['Y'] = df['positions_0_y']
    df['C'] = abs(df['positions_0_y'] - 50)

    x = df['X'].values*(105/100)
    y = df['C'].values*(65/100)
    df['Distance'] = np.sqrt(x**2 + y**2)
    df['Distance_root'] = np.sqrt(df['Distance'])
    a = np.arctan((7.32 *x) /(x**2 + y**2 - (7.32/2)**2))
    df['Angle'] = a
    df['Angle'] = np.where(df['Angle'] < 0, df['Angle'] + np.pi, df['Angle']) 
    df['Angle_deg'] = np.rad2deg(df['Angle'])
    return df

def filter_out_id(df_orig, id_col_name):
    df = df_orig.copy()
    df = df[df[id_col_name]==0].reset_index()
    return df

def make_data(filepath):
    shot_subEventName= 'Shot'
    goal_event_tag = 101
    header_event_tag = 403
    goal_col_name = 'goal'
    header_col_name = 'is_header'
    df = load_events(filepath)
    df = filter_on_subEvent(df, shot_subEventName)
    df = id_col_on_tag(df, goal_event_tag, goal_col_name)
    df = id_col_on_tag(df, header_event_tag, header_col_name)
    df = filter_out_id(df, header_col_name)
    df = make_geometry(df)
    df = get_grid_coords(df)
    return df

def stack_qcuts(df_orig, variables, prop_on_col='goal', n_quantiles=20):
    sub_dfs = []
    for var in variables:
        sub_df = df_orig.copy()
        sub_df['bin'] = pd.qcut(sub_df[var], n_quantiles)
        sub_df = (sub_df.groupby(by=['bin'])[prop_on_col].value_counts() / sub_df.groupby(by=['bin'])[prop_on_col].count()).to_frame()
        sub_df = sub_df.rename(columns={prop_on_col: 'prob_' + prop_on_col}).reset_index()
        sub_df[var] = sub_df['bin'].apply(lambda x: x.mid).astype(float)
        sub_df = sub_df[sub_df[prop_on_col]==1].reset_index(drop=True).reset_index()
        sub_df = sub_df.melt(id_vars=[col for col in sub_df.columns if col != var], value_name='midpoint')
        sub_dfs.append(sub_df)
    df = pd.concat(sub_dfs, axis=0).reset_index(drop=True)
    return df 

def calc_dist(x, y):
    return np.sqrt(x**2 + abs(y-65/2)**2)

def calc_angle(x, y):
    a = np.arctan(7.32 *x /(x**2 + abs(y-65/2)**2 - (7.32/2)**2))
    # a = np.arctan((7.32 *x) /(x**2 + y**2 - (7.32/2)**2))
    a = np.where(a < 0, a + np.pi, a) 
    a =  np.rad2deg(a)
    return a