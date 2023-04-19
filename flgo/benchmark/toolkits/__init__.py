import gzip

import numpy as np

from flgo.benchmark.base import BasicTaskPipe
from flgo.benchmark.base import BasicTaskGenerator
from flgo.benchmark.base import BasicTaskCalculator
import urllib.request
import zipfile
import os


def download_from_url(url=None, filepath='.'):
    """Download dataset from url to filepath."""
    if url:urllib.request.urlretrieve(url, filepath)
    return filepath


def extract_from_zip(src_path, target_path, pwd=None):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path, pwd=pwd)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]

def extract_one_from_zip(src_path, target_path, file_name, pwd=None):
    f = zipfile.ZipFile(src_path)
    f.extract(file_name, target_path, pwd=pwd)
    f.close()



def extract_from_gz(src_file, target_file):
    """Unzip the .gz file (src_path) to target_path"""
    with open(target_file, 'wb') as f:
        zf = gzip.open(src_file, mode='rb')
        f.write(zf.read())
        zf.close()
    return target_file


def normalized(rawdata, normalize):
    n, m = rawdata.shape
    scale = np.ones(m)
    if normalize == 0:
        data = rawdata
    elif normalize == 1:
        data = rawdata / np.max(rawdata)
    elif normalize == 2:
        data = np.zeros((n, m))
        for i in range(m):
            scale[i] = np.max(np.abs(rawdata[:, i]))
            data[:, i] = rawdata[:, i] / np.max(np.abs(rawdata[:, i]))
    else:
        raise RuntimeError("The parameter 'normalize' can only take values from 0, 1, 2")
    return data