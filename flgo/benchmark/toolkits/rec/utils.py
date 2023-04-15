import pandas as pd

def map_column(data:pd.DataFrame, col_name='', map_list = None):
    """reID  the value """
    if col_name not in data: raise KeyError('key {} is not in columns of data')
    all_samples = data[col_name].unique()
    if map_list is None: map_list = list(range(len(all_samples)))
    sample_map = {sid: sidx for sidx, sid in zip(map_list, all_samples)}
    return data[col_name].map(sample_map)
