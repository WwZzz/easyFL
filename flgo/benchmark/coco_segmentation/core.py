import os

import torch
from torchvision.datasets.utils import download_and_extract_archive
import flgo
import torchvision
from flgo.benchmark.toolkits.cv.segmentation import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator
from ..toolkits.cv.segmentation.utils import get_transform
from ..toolkits.cv.segmentation.coco_utils import FilterAndRemapCocoCategories, ConvertCocoPolysToMask
from ..toolkits.cv.segmentation.transforms import Compose
from flgo.benchmark.toolkits.cv.segmentation.coco_utils import _coco_remove_images_without_annotations
from ..toolkits import download_from_url, extract_from_zip

CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
PATHS = {
        "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
        "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
        # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    }
builtin_class = torchvision.datasets.CocoDetection
train_transforms = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask(), get_transform(train=True)])
test_transforms = Compose([FilterAndRemapCocoCategories(CAT_LIST, remap=True), ConvertCocoPolysToMask(), get_transform(train=False)])
path = os.path.join(flgo.benchmark.path, 'RAW_DATA', 'COCO')

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=path):
        super(TaskGenerator, self).__init__(benchmark=os.path.split(os.path.dirname(__file__))[-1], rawdata_path=rawdata_path,
                                            builtin_class=builtin_class, train_transform=train_transforms, test_transform=test_transforms, num_classes = 21)
        self.additional_option = {}
        self.train_image_path = os.path.join(self.rawdata_path, "train2017")
        self.train_annotation_path = os.path.join(self.rawdata_path,"annotations", "instances_train2017.json")
        self.val_image_path = os.path.join(self.rawdata_path, "val2017")
        self.val_annotation_path = os.path.join(self.rawdata_path,"annotations", "instances_val2017.json")
        self.train_additional_option = {'root':self.train_image_path, 'annFile':self.train_annotation_path}
        self.test_additional_option = {'root':self.val_image_path, 'annFile':self.val_annotation_path}
        self.num_classes = 21

    def load_data(self):
        # download the dataset if the dataset doesn't exist
        if not os.path.exists(self.train_image_path):
            self.download_data('train')
        if not os.path.exists(self.val_image_path):
            self.download_data('val')
        if not os.path.exists(os.path.join(self.rawdata_path, "annotations")):
            self.download_data('anno')
        # load the datasets
        train_default_init_para = {'root': self.rawdata_path, 'download':self.download_data, 'train':True, 'transforms':self.train_transform}
        test_default_init_para = {'root': self.rawdata_path, 'download':self.download_data, 'train':False, 'transforms':self.test_transform}
        train_default_init_para.update(self.additional_option)
        train_default_init_para.update(self.train_additional_option)
        test_default_init_para.update(self.additional_option)
        test_default_init_para.update(self.test_additional_option)
        train_pop_key = [k for k in train_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
        test_pop_key = [k for k in test_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
        for k in train_pop_key: train_default_init_para.pop(k)
        for k in test_pop_key: test_default_init_para.pop(k)
        # init datasets
        self.train_data = self.builtin_class(**train_default_init_para)
        self.test_data = self.builtin_class(**test_default_init_para)
        self.train_data = _coco_remove_images_without_annotations(self.train_data, CAT_LIST)

    def download_data(self, flag:str):
        if flag=='train':
            url = "http://images.cocodataset.org/zips/train2017.zip"
            filename = "train2017.zip"
        elif flag=='val':
            url = "http://images.cocodataset.org/zips/val2017.zip"
            filename = "val2017.zip"
        elif flag=='anno':
            url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
            filename = "annotations_trainval2017.zip"
        download_and_extract_archive(url, download_root=self.rawdata_path, filename=filename)
        return

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class, train_transforms, test_transforms)

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_default_init_para = {'root': self.feddata['rawdata_path'], 'download':True, 'train':True, 'transforms':self.train_transform}
        test_default_init_para = {'root': self.feddata['rawdata_path'], 'download':True, 'train':False, 'transforms':self.test_transform}
        if 'additional_option' in self.feddata.keys():
            train_default_init_para.update(self.feddata['additional_option'])
            test_default_init_para.update(self.feddata['additional_option'])
        if 'train_additional_option' in self.feddata.keys(): train_default_init_para.update(self.feddata['train_additional_option'])
        if 'test_additional_option' in self.feddata.keys(): test_default_init_para.update(self.feddata['test_additional_option'])
        train_pop_key = [k for k in train_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
        test_pop_key = [k for k in test_default_init_para.keys() if k not in self.builtin_class.__init__.__annotations__]
        for k in train_pop_key: train_default_init_para.pop(k)
        for k in test_pop_key: test_default_init_para.pop(k)
        train_data = self.builtin_class(**train_default_init_para)
        train_data = _coco_remove_images_without_annotations(train_data, CAT_LIST)
        test_data = self.builtin_class(**test_default_init_para)
        test_data = self.TaskDataset(test_data, list(range(len(test_data))), None, running_time_option['pin_memory'])
        # rearrange data for server
        server_data_test, server_data_val = self.split_dataset(test_data, running_time_option['test_holdout'])
        num_classes = self.feddata['num_classes']
        if server_data_val is not None: server_data_val.num_classes = num_classes
        if server_data_test is not None: server_data_test.num_classes = num_classes
        task_data = {'server': {'test': server_data_test, 'val': server_data_val}}
        # rearrange data for clients
        local_perturbation = self.feddata['local_perturbation'] if 'local_perturbation' in self.feddata.keys() else [None for _ in self.feddata['client_names']]
        for cid, cname in enumerate(self.feddata['client_names']):
            cpert = None if  local_perturbation[cid] is None else [torch.tensor(t) for t in local_perturbation[cid]]
            cdata = self.TaskDataset(train_data, self.feddata[cname]['data'], cpert, running_time_option['pin_memory'])
            cdata_train, cdata_val = self.split_dataset(cdata, running_time_option['train_holdout'])
            if running_time_option['train_holdout']>0 and running_time_option['local_test']:
                cdata_val, cdata_test = self.split_dataset(cdata_val, 0.5)
            else:
                cdata_test = None
            if cdata_train is not None: cdata_train.num_classes = num_classes
            if cdata_val is not None: cdata_val.num_classes = num_classes
            if cdata_test is not None: cdata_test.num_classes = num_classes
            task_data[cname] = {'train':cdata_train, 'val':cdata_val, 'test': cdata_test}
        return task_data

TaskCalculator = GeneralCalculator