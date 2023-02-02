import random
import paddle
import ujson
from benchmark.toolkits.base import *

class BuiltinClassGenerator(BasicTaskGenerator):
    def __init__(self, benchmark, rawdata_path, builtin_class, transform=None, is_built=True):
        super(BuiltinClassGenerator, self).__init__(benchmark, rawdata_path)
        self.builtin_class = builtin_class
        self.transform = transform
        self.additional_option = {}
        self.is_built = is_built

    def load_data(self):
        if self.is_built:
            self.train_data = self.builtin_class(download=True, mode='train', transform=self.transform)
            self.test_data = self.builtin_class(download=True, mode='test', transform=self.transform)
        else:
            self.train_data = self.builtin_class(image_path=self.rawdata_path, label_path=self.rawdata_path,
                                                 download=True, mode='train', transform=self.transform)
            self.test_data = self.builtin_class(image_path=self.rawdata_path, label_path=self.rawdata_path,
                                                download=True, mode='test', transform=self.transform)

    def partition(self):
        self.local_datas = self.partitioner(self.train_data)

class BuiltinClassPipe(HorizontalTaskPipe):
    class TaskDataset(paddle.io.Subset):
        def __init__(self, dataset, indices, perturbation=None):
            super().__init__(dataset, indices)
            self.dataset = dataset
            self.indices = indices
            self.perturbation = {idx:p for idx, p in zip(indices, perturbation)} if perturbation is not None else None

        def __getitem__(self, idx):
            if self.perturbation is None:
                if isinstance(idx, list):
                    return self.dataset[[self.indices[i] for i in idx]]
                return self.dataset[self.indices[idx]]
            else:
                return self.dataset[self.indices[idx]][0] + self.perturbation[self.indices[idx]],  self.dataset[self.indices[idx]][1]

    def __init__(self, task_name, buildin_class, transform=None, is_built=True):
        super(BuiltinClassPipe, self).__init__(task_name)
        self.builtin_class = buildin_class
        self.is_built = is_built
        self.transform = transform

    def save_task(self, generator):
        client_names = self.gen_client_names(len(generator.local_datas))
        feddata = {'client_names': client_names, 'server_data': list(range(len(generator.test_data))),  'rawdata_path': generator.rawdata_path, 'additional_option': generator.additional_option}
        for cid in range(len(client_names)): feddata[client_names[cid]] = {'data': generator.local_datas[cid],}
        if hasattr(generator.partitioner, 'local_perturbation'): feddata['local_perturbation'] = generator.partitioner.local_perturbation
        with open(os.path.join(self.task_path, 'data.json'), 'w') as outf:
            ujson.dump(feddata, outf)
        return

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        if self.is_built:
            train_data = self.builtin_class(download=True, mode='train', transform=self.transform,
                                            **self.feddata['additional_option'])
            test_data = self.builtin_class(download=True, mode='test', transform=self.transform,
                                           **self.feddata['additional_option'])
        else:
            train_data = self.builtin_class(
                image_path=self.feddata['rawdata_path'], label_path=self.feddata['rawdata_path'], download=True,
                mode='train', transform=self.transform, **self.feddata['additional_option'])
            test_data = self.builtin_class(
                image_path=self.feddata['rawdata_path'], label_path=self.feddata['rawdata_path'], download=True,
                mode='test', transform=self.transform, **self.feddata['additional_option'])
       # rearrange data for server
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if local_perturbation[cid] is None else [paddle.to_tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], cpert)
            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            task_data[cname] = {'train':cdata_train, 'valid':cdata_valid}
        return task_data

class GeneralCalculator(BasicTaskCalculator):
    def __init__(self, device, optimizer_name='SGD'):
        super(GeneralCalculator, self).__init__(device, optimizer_name)
        self.criterion = paddle.nn.CrossEntropyLoss()
        self.DataLoader = paddle.io.DataLoader

    def compute_loss(self, model, data):
        """
        :param model: the model to train
        :param data: the training dataset
        :return: dict of train-one-step's result, which should at least contains the key 'loss'
        """
        # tdata = self.to_device(data)
        outputs = model(data[0])
        loss = self.criterion(outputs, data[-1])
        return {'loss': loss}

    @paddle.no_grad()
    def test(self, model, dataset, batch_size=64, num_workers=0):
        """
        Metric = [mean_accuracy, mean_loss]
        :param model:
        :param dataset:
        :param batch_size:
        :return: [mean_accuracy, mean_loss]
        """
        model.eval()
        if batch_size==-1:batch_size=len(dataset)
        data_loader = self.get_dataloader(dataset, batch_size=batch_size, num_workers=num_workers)
        total_loss = 0.0
        num_correct = 0
        for batch_id, batch_data in enumerate(data_loader):
            # batch_data = self.to_device(batch_data)
            outputs = model(batch_data[0])
            batch_mean_loss = self.criterion(outputs, batch_data[-1]).item()
            y_pred = outputs.argmax(1, keepdim=True)
            correct = y_pred.equal(batch_data[-1].reshape(y_pred.shape)).cast('long').sum()
            num_correct += correct.item()
            total_loss += batch_mean_loss * len(batch_data[-1])
        return {'accuracy': 1.0*num_correct/len(dataset), 'loss':total_loss/len(dataset)}

    def to_device(self, data):
        return data[0].cuda(self.device), data[1].cuda(self.device)

    def get_dataloader(self, dataset, batch_size=64, shuffle=True, num_workers=0):
        if self.DataLoader == None:
            raise NotImplementedError("DataLoader Not Found.")
        return self.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

class GeneralGenerator(BasicTaskGenerator):
    def __init__(self, benchmark, rawdata_path):
        super(GeneralGenerator, self).__init__(benchmark, rawdata_path)
        return

    def load_data(self):
        pass

    def partition(self):
        return self.partitioner(self.train_data)

