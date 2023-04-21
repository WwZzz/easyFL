import pandas as pd


names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marial-status', 'occupation', 'relationship',
             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
column_type = ['continuous', 8, 'continuous', 16, 'continuous', 7, 14, 6, 5, 2, 'continuous', 'continuous',
                   'continuous', 41, 2]

train = pd.read_csv('adult.data', skipinitialspace=True, names=names)
test = pd.read_csv('adult.test', skipinitialspace=True, names=names)

