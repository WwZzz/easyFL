import os
import shutil
from typing import Any, Callable, Optional, Tuple
import numpy as np
import torch
from PIL import Image
from torchvision.datasets.utils import download_url, verify_str_arg, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
from flgo.benchmark.toolkits.cv.segmentation import BuiltinClassGenerator, BuiltinClassPipe, GeneralCalculator


class SBDataset(VisionDataset):
    """`Semantic Boundaries Dataset <http://home.bharathh.info/pubs/codes/SBD/download.html>`_

    The SBD currently contains annotations from 11355 images taken from the PASCAL VOC 2011 dataset.

    .. note ::

        Please note that the train and val splits included with this dataset are different from
        the splits in the PASCAL VOC dataset. In particular some "train" images might be part of
        VOC2012 val.
        If you are interested in testing on VOC 2012 val, then use `image_set='train_noval'`,
        which excludes all val images.

    .. warning::

        This class needs `scipy <https://docs.scipy.org/doc/>`_ to load target files from `.mat` format.

    Args:
        root (string): Root directory of the Semantic Boundaries Dataset
        image_set (string, optional): Select the image_set to use, ``train``, ``val`` or ``train_noval``.
            Image set ``train_noval`` excludes VOC 2012 val images.
        mode (string, optional): Select target type. Possible values 'boundaries' or 'segmentation'.
            In case of 'boundaries', the target is an array of shape `[num_classes, H, W]`,
            where `num_classes=20`.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version. Input sample is PIL image and target is a numpy array
            if `mode='boundaries'` or PIL image if `mode='segmentation'`.
    """

    url = "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz"
    md5 = "82b4d87ceb2ed10f6038a1cba92111cb"
    filename = "benchmark.tgz"

    voc_train_url = "http://home.bharathh.info/pubs/codes/SBD/train_noval.txt"
    voc_split_filename = "train_noval.txt"
    voc_split_md5 = "79bff800c5f0b1ec6b21080a3c066722"

    def __init__(
            self,
            root: str,
            image_set: str = "train",
            mode: str = "boundaries",
            download: bool = False,
            transforms: Optional[Callable] = None,
    ) -> None:

        try:
            from scipy.io import loadmat

            self._loadmat = loadmat
        except ImportError:
            raise RuntimeError("Scipy is not found. This dataset needs to have scipy installed: pip install scipy")

        super().__init__(root, transforms)
        self.image_set = verify_str_arg(image_set, "image_set", ("train", "val", "train_noval"))
        self.mode = verify_str_arg(mode, "mode", ("segmentation", "boundaries"))
        self.num_classes = 20

        sbd_root = self.root
        image_dir = os.path.join(sbd_root, "img")
        mask_dir = os.path.join(sbd_root, "cls")

        if download and not os.path.exists(os.path.join(self.root, "benchmark_RELEASE", "dataset")):
            download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.md5)
            extracted_ds_root = os.path.join(self.root, "benchmark_RELEASE", "dataset")
            for f in ["cls", "img", "inst", "train.txt", "val.txt"]:
                old_path = os.path.join(extracted_ds_root, f)
                shutil.move(old_path, sbd_root)
            download_url(self.voc_train_url, sbd_root, self.voc_split_filename, self.voc_split_md5)

        if not os.path.isdir(sbd_root):
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        split_f = os.path.join(sbd_root, image_set.rstrip("\n") + ".txt")

        with open(os.path.join(split_f)) as fh:
            file_names = [x.strip() for x in fh.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks = [os.path.join(mask_dir, x + ".mat") for x in file_names]

        self._get_target = self._get_segmentation_target if self.mode == "segmentation" else self._get_boundaries_target

    def _get_segmentation_target(self, filepath: str) -> Image.Image:
        mat = self._loadmat(filepath)
        return Image.fromarray(mat["GTcls"][0]["Segmentation"][0])

    def _get_boundaries_target(self, filepath: str) -> np.ndarray:
        mat = self._loadmat(filepath)
        return np.concatenate(
            [np.expand_dims(mat["GTcls"][0]["Boundaries"][0][i][0].toarray(), axis=0) for i in range(self.num_classes)],
            axis=0,
        )

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img = Image.open(self.images[index]).convert("RGB")
        target = self._get_target(self.masks[index])

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self) -> int:
        return len(self.images)

    def extra_repr(self) -> str:
        lines = ["Image set: {image_set}", "Mode: {mode}"]
        return "\n".join(lines).format(**self.__dict__)


builtin_class = SBDataset
transforms = None

class TaskGenerator(BuiltinClassGenerator):
    def __init__(self, rawdata_path=os.path.join(flgo.benchmark.path, 'RAW_DATA', 'SBDATASET')):
        super(TaskGenerator, self).__init__(benchmark='sbdataset_segmentation', rawdata_path=rawdata_path,
                                            builtin_class=builtin_class, transform=transforms)
        self.num_classes = 21
        self.additional_option = {'mode':'segmentation'}
        self.train_additional_option = {'image_set':'train_noval'}
        self.test_additional_option = {'image_set':'val'}

    def load_data(self):
        # load the datasets
        train_default_init_para = {'root': self.rawdata_path, 'download': self.download, 'train': True,
                                   'transforms': self.transform}
        test_default_init_para = {'root': self.rawdata_path, 'download': self.download, 'train': False,
                                  'transforms': self.transform}
        train_default_init_para.update(self.additional_option)
        train_default_init_para.update(self.train_additional_option)
        test_default_init_para.update(self.additional_option)
        test_default_init_para.update(self.test_additional_option)
        train_pop_key = [k for k in train_default_init_para.keys() if
                         k not in self.builtin_class.__init__.__annotations__]
        test_pop_key = [k for k in test_default_init_para.keys() if
                        k not in self.builtin_class.__init__.__annotations__]
        for k in train_pop_key: train_default_init_para.pop(k)
        for k in test_pop_key: test_default_init_para.pop(k)
        # init datasets
        self.train_data = self.builtin_class(**train_default_init_para)
        self.test_data = self.builtin_class(**test_default_init_para)

class TaskPipe(BuiltinClassPipe):
    def __init__(self, task_path):
        super(TaskPipe, self).__init__(task_path, builtin_class, transforms)

    def load_data(self, running_time_option) -> dict:
        # load the datasets
        train_default_init_para = {'root': self.feddata['rawdata_path'], 'download':True, 'train':True, 'transforms':self.transform}
        test_default_init_para = {'root': self.feddata['rawdata_path'], 'download':True, 'train':False, 'transforms':self.transform}
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