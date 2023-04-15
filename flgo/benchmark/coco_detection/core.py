import sys
from typing import Any

sys.path.append('/home/wz/anaconda3/lib/python3.9/site-packages/')
import os
import torch
from torchvision.datasets.utils import download_and_extract_archive
import flgo
import torchvision
from flgo.benchmark.toolkits.cv.detection import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
from ..toolkits.cv.detection.utils import get_transform
from ..toolkits.cv.detection.coco_utils import ConvertCocoPolysToMask
from ..toolkits.cv.detection.transforms import Compose

CAT_LIST = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80]
PATHS = {
        "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
}

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, root:str, annFile:str, transforms:Any):
        super().__init__(root, annFile)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

builtin_class = CocoDetection
train_transform = Compose([ConvertCocoPolysToMask(), get_transform(train=True)])
test_transform = Compose([ConvertCocoPolysToMask(), get_transform(train=False)])
path = os.path.join(flgo.benchmark.path, 'RAW_DATA', 'COCO')

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1], rawdata_path=rawdata_path,
                                            builtin_class=builtin_class)
        self.num_classes = 91
        self.test_transform = test_transform
        self.train_transform = train_transform
        self.additional_option = {}
        self.train_image_path = os.path.join(self.rawdata_path, "train2017")
        self.train_annotation_path = os.path.join(self.rawdata_path,"annotations", "instances_train2017.json")
        self.val_image_path = os.path.join(self.rawdata_path, "val2017")
        self.val_annotation_path = os.path.join(self.rawdata_path,"annotations", "instances_val2017.json")
        self.train_additional_option = {'root':self.train_image_path,'image_set':'train', 'annFile':self.train_annotation_path}
        self.test_additional_option = {'root':self.val_image_path,'image_set':'val', 'annFile':self.val_annotation_path}

    def load_data(self):
        # load the datasets
        train_default_init_para = {'transforms':self.train_transform}
        test_default_init_para = {'transforms':self.test_transform}
        train_default_init_para.update(self.additional_option)
        train_default_init_para.update(self.train_additional_option)
        test_default_init_para.update(self.additional_option)
        test_default_init_para.update(self.test_additional_option)
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            train_pop_key = [k for k in train_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            test_pop_key = [k for k in test_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in train_pop_key: train_default_init_para.pop(k)
            for k in test_pop_key: test_default_init_para.pop(k)
        # init datasets
        self.train_data = self.builtin_class(**train_default_init_para)
        self.test_data = self.builtin_class(**test_default_init_para)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class)
        self.test_transform = test_transform
        self.train_transform = train_transform

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_default_init_para = {'transforms':self.train_transform}
        test_default_init_para = {'transforms':self.test_transform}
        if 'additional_option' in self.feddata.keys():
            train_default_init_para.update(self.feddata['additional_option'])
            test_default_init_para.update(self.feddata['additional_option'])
        if 'train_additional_option' in self.feddata.keys(): train_default_init_para.update(self.feddata['train_additional_option'])
        if 'test_additional_option' in self.feddata.keys(): test_default_init_para.update(self.feddata['test_additional_option'])
        if 'kwargs' not in self.builtin_class.__init__.__annotations__:
            train_pop_key = [k for k in train_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            test_pop_key = [k for k in test_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
            for k in train_pop_key: train_default_init_para.pop(k)
            for k in test_pop_key: test_default_init_para.pop(k)
        train_data = self.builtin_class(**train_default_init_para)
        test_data = self.builtin_class(**test_default_init_para)
        test_data = self.TaskDataset(test_data, list(range(len(test_data))), None, running_time_option['pin_memory'])
        # rearrange data for server
        server_data_test, server_data_valid = self.split_dataset(test_data, running_time_option['test_holdout'])
        task_data = {'server': {'test': server_data_test, 'valid': server_data_valid}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if  local_perturbation[cid] is None else [torch.tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], cpert, running_time_option['pin_memory'])
            cdata_train, cdata_valid = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_valid, cdata_test = self.split_dataset(cdata_valid, 0.5)
            else:
                cdata_test = None
            task_data[cname] = {'train':cdata_train, 'valid':cdata_valid, 'test': cdata_test}
        return task_data

TaskCalculator = GeneralCalculator