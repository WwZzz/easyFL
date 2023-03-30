import torchvision
from flgo.benchmark.toolkits.cv.horizontal.image_classification import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
import flgo.benchmark
import os.path
transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

TaskCalculator = GeneralCalculator

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path,'RAW_DATA', 'SVHN')):
        super(TaskGenerator, self).__init__('svhn_classification', rawdata_path, torchvision.datasets.SVHN, transforms)

    def load_data(self):
        self.train_data = self.builtin_class(root=self.rawdata_path, download=True, split='train', transform=self.transform)
        self.test_data = self.builtin_class(root=self.rawdata_path, download=True, split='test',transform=self.transform)


class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_name):
        super(TaskPipe, self).__init__(task_name, torchvision.datasets.SVHN, transforms)

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_data = self.builtin_class(root=self.feddata['rawdata_path'], download=True, split = 'train', transform=self.transform, **self.feddata['additional_option'])
        test_data = self.builtin_class(root=self.feddata['rawdata_path'], download=True, split = 'test', transform=self.transform, **self.feddata['additional_option'])
        # rearrange data for server
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if  local_perturbation[cid] is None else [torch.tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], cpert)
            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train':cdata_train, 'valid':cdata_valid}
        return task_data