from flgo.benchmark.base import BasicTaskPipe
from flgo.benchmark.base import BasicTaskGenerator
from flgo.benchmark.base import BasicTaskCalculator
import urllib.request
import zipfile
import os

def download_from_url(url= None, filepath = '.'):
    """Download dataset from url to filepath."""
    if url:urllib.request.urlretrieve(url, filepath)
    return filepath

def extract_from_zip(src_path, target_path):
    """Unzip the .zip file (src_path) to target_path"""
    f = zipfile.ZipFile(src_path)
    f.extractall(target_path)
    targets = f.namelist()
    f.close()
    return [os.path.join(target_path, tar) for tar in targets]