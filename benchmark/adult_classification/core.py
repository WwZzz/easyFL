"""This benchmark is from https://github.com/litian96/fair_flearn/tree/master/data/adult"""
from benchmark.toolkits import DefaultTaskGen

from torchvision import datasets, transforms
from benchmark.toolkits import ClassificationCalculator as TaskCalculator
from benchmark.toolkits import IDXTaskPipe as TaskPipe
from benchmark.toolkits import DefaultTaskGen
from torch.utils.data import Dataset
import os, urllib
import numpy as np
import torch


class TaskGen(DefaultTaskGen):
    def __init__(self, dist_id, num_clients = 1, skewness = 0.5, local_hld_rate=0.2, seed=0):
        super(TaskGen, self).__init__(benchmark='adult_classification',
                                      dist_id=dist_id,
                                      num_clients=num_clients,
                                      skewness=skewness,
                                      rawdata_path='./benchmark/RAW_DATA/ADULT',
                                      local_hld_rate=local_hld_rate,
                                      seed=seed
                                      )
        self.num_classes = 2
        self.num_clients = 2
        self.cnames = self.get_client_names()
        self.dist = 5
        self.skewness = 1.0
        self.taskname = self.get_taskname()
        self.taskpath = os.path.join(self.task_rootpath, self.taskname)
        self.save_task = TaskPipe.save_task
        self.source_dict = {
            'class_path': 'benchmark.adult_classification.core',
            'class_name': 'Adult',
            'train_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'train': 'True'
            },
            'test_args': {
                'root': '"'+self.rawdata_path+'"',
                'download': 'True',
                'train': 'False'
            }
        }

    def load_data(self):
        self.train_data = Adult(self.rawdata_path, train=True)
        self.test_data = Adult(self.rawdata_path, train=False)

    def mark_data(self, data, mark_func):
        return [mark_func(di[0]) for di in data]

    def partition(self):
        marks = self.mark_data(self.train_data, lambda x:x[21])
        didxs = list(range(len(self.train_data)))
        local_datas = [[],[]]
        for did in didxs:
            local_datas[int(marks[did])].append(did)
        return local_datas


def isFloat(string):
    # credits: http://stackoverflow.com/questions/2356925/how-to-check-whether-string-might-be-type-cast-to-float-in-python
    try:
        float(string)
        return True
    except ValueError:
        return False

def find_means_for_continuous_types(X):
    means = []
    for col in range(len(X[0])):
        summ = 0
        count = 0.000000000000000000001
        for value in X[:, col]:
            if isFloat(value):
                summ += float(value)
                count +=1
        means.append(summ/count)
    return means



class Adult(Dataset):
    resources = [
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data',
        'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test',
    ]

    training_file = "adult.data"
    test_file = "adult.test"
    classes = ['0 - <=50K', '1 - >50K']
    inputs = (
        ("age", ("continuous",)),
        ("workclass", (
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay",
        "Never-worked")),
        ("fnlwgt", ("continuous",)),
        ("education", (
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc", "9th", "7th-8th",
        "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool")),
        ("education-num", ("continuous",)),
        ("marital-status", (
        "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent",
        "Married-AF-spouse")),
        ("occupation", ("Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                        "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                        "Priv-house-serv", "Protective-serv", "Armed-Forces")),
        ("relationship", ("Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried")),
        ("race", ("White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black")),
        ("sex", ("Female", "Male")),
        ("capital-gain", ("continuous",)),
        ("capital-loss", ("continuous",)),
        ("hours-per-week", ("continuous",)),
        ("native-country", (
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)",
        "India", "Japan", "Greece", "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland",
        "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic", "Laos", "Ecuador",
        "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand", "Yugoslavia",
        "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"))
    )

    def __init__(self, root='.', train=True, download=True):
        self.root = root
        self.train_path = os.path.join(self.root, self.training_file)
        self.test_path = os.path.join(self.root, self.test_file)
        self.train = train
        if download==True or (not os.path.exists(self.train_path)) or (not os.path.exists(self.test_path)):
            self.download()
        if self.train==True:
            x,y = self.generate_dataset(self.train_path)
        else:
            x,y = self.generate_dataset(self.test_path)
        self.data = torch.concat((torch.tensor(x), torch.tensor(y).unsqueeze(1)), dim=1)

    def download(self):
        def download_from_url(url=None, filepath='.'):
            """Download dataset from url to filepath."""
            if url: urllib.request.urlretrieve(url, filepath)
            return filepath
        if not os.path.exists(self.train_path):
            download_from_url(self.resources[0], self.train_path)
        if not os.path.exists(self.test_path):
            download_from_url(self.resources[1], self.test_path)
            with open(self.test_path, 'r') as f:
                lines = f.readlines()
                lines = lines[1:]
            with open(self.test_path, 'w') as df:
                df.writelines(lines)

    def generate_dataset(self, file_path):
        data = np.genfromtxt(file_path, delimiter=', ', dtype=str, autostrip=True)
        means = find_means_for_continuous_types(data)
        X, y = self.prepare_data(data, means)
        return X.tolist(), y.tolist()

    def prepare_data(self, raw_data, means):
        X = raw_data[:, :-1]
        y = raw_data[:, -1:]
        # X:
        def flatten_persons_inputs_for_model(person_inputs, means):
            input_shape = [1, 8, 1, 16, 1, 7, 14, 6, 5, 2, 1, 1, 1, 41]
            float_inputs = []
            for i in range(len(input_shape)):
                features_of_this_type = input_shape[i]
                is_feature_continuous = features_of_this_type == 1
                if is_feature_continuous:
                    # in order to be consistant with the google paper -- only train with categorical features
                    '''
                    mean = means[i]
                    if isFloat(person_inputs[i]):
                        scale_factor = 1/(2*mean)  # we prefer inputs mainly scaled from -1 to 1. 
                        float_inputs.append(float(person_inputs[i])*scale_factor)
                    else:
                        float_inputs.append(mean)
                    '''
                    pass
                else:
                    for j in range(features_of_this_type):
                        feature_name = self.inputs[i][1][j]

                        if feature_name == person_inputs[i]:
                            float_inputs.append(1.)
                        else:
                            float_inputs.append(0)
            return float_inputs
        new_X = []
        for person in range(len(X)):
            formatted_X = flatten_persons_inputs_for_model(X[person], means)
            new_X.append(formatted_X)
        new_X = np.array(new_X)
        new_y = []
        for i in range(len(y)):
            if y[i] == ">50K" or y[i] == ">50K.":
                new_y.append(1)
            else:  # y[i] == "<=50k":
                new_y.append(0)

        new_y = np.array(new_y)

        return (new_X, new_y)

    def __getitem__(self, item):
        return self.data[item][:-1], torch.tensor(self.data[item][-1], dtype=torch.long)

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':
    # a = Adult('../RAW_DATA/ADULT', train=True)
    a = Adult('../RAW_DATA/ADULT', train=False)
    print('ok')
