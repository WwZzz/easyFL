import collections
try:
    import resource
except:
    resource = None
from itertools import chain
try:
    import ujson as json
except:
    import json
import os
import uuid
import flgo.utils.fflow as fuf
import torch
import numpy as np
import torch.utils.data as tud
import sys
import pickle
from tqdm import tqdm

class MemmapManager:
    def __init__(self, root:str, num_elems: int=-1):
        self.root = root
        self.num_blocks = 0
        self.crt_block_id = 0
        self.crt_block_size = 0
        self.data_block = collections.defaultdict(list)
        self.num_elems = num_elems
        self.item_shape = [None for _ in range(num_elems)]
        self.data_meta = self.load_meta() if os.path.exists(os.path.join(self.root, 'meta.json')) else {}
        self.block_map = [[]]
        self.block_buffer = {}

    def add(self, data, name:str):
        assert len(data)==self.num_elems
        data_meta = {
            'num_elems': self.num_elems,
            'block_id': self.crt_block_id,
            'size': [],
        }
        for i in range(self.num_elems):
            self.data_block[i].append(data[i])
            data_meta['size'].append(len(data[i]) if data[i] is not None else 0)
        self.data_meta[name] = data_meta
        self.block_map[self.crt_block_id].append(name)
        self.crt_block_size += 1

    def clear_block(self):
        self.data_block = collections.defaultdict(list)
        self.block_map.append([])
        self.num_blocks += 1
        self.crt_block_id += 1
        self.crt_block_size += 1

    def is_empty(self):
        return len(list(self.data_block.keys()))==0

    def dump(self):
        block_name = f"db{self.crt_block_id}"
        data_names = self.block_map[self.crt_block_id]
        for dname in data_names:
            self.data_meta[dname]['index'] = []
            self.data_meta[dname]['block_files'] = []
        for elem_id in range(self.num_elems):
            file_name = os.path.join(self.root, '.'.join([block_name, str(elem_id), 'npy']))
            try:
                data_to_save = np.concatenate(self.data_block[elem_id], axis=0)
                sizes = [0]+[self.data_meta[dname]['size'][elem_id] for dname in data_names]
                indices = np.cumsum(sizes).astype(np.int32).tolist()
                for d_idx,dname in zip(indices, data_names):
                    self.data_meta[dname]['index'].append(d_idx)
            except ValueError as e:
                # flatten
                shapes = [tuple(d.shape) for d in self.data_block[elem_id]]
                data_to_save = [di.ravel() for di in self.data_block[elem_id]]
                sizes = [0]+[len(di) for di in data_to_save]
                indices = np.cumsum(sizes).astype(np.int32).tolist()
                data_to_save = np.concatenate(data_to_save)
                sizes = sizes[1:]
                self.data_meta[dname]['shape'] = []
                for dshape, dsize, d_idx, dname in zip(shapes, sizes, indices, data_names):
                    self.data_meta[dname]['index'].append(d_idx)
                    self.data_meta[dname]['shape'].append(dshape)
                    self.data_meta[dname]['size'][elem_id] = dsize
            for dname in data_names: self.data_meta[dname]['block_files'].append(file_name)
            np.save(file_name, data_to_save, allow_pickle=True)
        self.clear_block()

    def save_meta(self):
        meta_file = os.path.join(self.root, 'meta.json')
        with open(meta_file, 'w') as f:
            json.dump(self.data_meta, f)

    def load_meta(self):
        meta_file = os.path.join(self.root, 'meta.json')
        with open(meta_file, 'r') as f:
            data_meta = json.load(f)
        return data_meta

    def get(self, data_name):
        dmeta = self.data_meta[data_name]
        num_elems, block_files, indices, sizes = dmeta['num_elems'], dmeta['block_files'], dmeta['index'], dmeta['size']
        for block in block_files:
            if block not in self.block_buffer:
                self.block_buffer[block] = np.load(block, mmap_mode='r', allow_pickle=True)
        shapes = dmeta.get('shape', None)
        sharable_data = [self.block_buffer[block_files[i]][indices[i]: indices[i]+sizes[i]] for i in range(num_elems)]
        if shapes is not None: sharable_data = [sharable_data[i].reshape(shapes[i]) for i in range(num_elems)]
        return sharable_data

def get_dict_size(obj, seen=None):
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        for key, value in obj.items():
            size += sys.getsizeof(key) + get_dict_size(value, seen)
    elif isinstance(obj, (list, tuple, set)):
        for item in obj:
            size += get_dict_size(item, seen)
    return size

TYPE_CANDIDATES = ['int', 'float', 'str', 'ndarray', 'Tensor', 'list', 'tuple', 'dict', 'int64', 'float64']

class MemDataset(tud.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = [d for d in data]

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class TmpDataset(tud.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, i):
        return tuple(d[i] for d in self.data)

class TmpDictDataset(tud.Dataset):
    def __init__(self, all_keys, data):
        super().__init__()
        self.data = data
        self.all_keys = all_keys

    def __len__(self):
        return len(self.data[0])

    def __getitem__(self, i):
        return unflatten_dict({k:v[i] for k,v in zip(self.all_keys, self.data)})

def _check_vector_shapes(vec_list):
    """
    Check whether the tensors\ndarrays have the same shape

    Args:
        vec_list (list[torch.Tensor]|list[numpy.ndarray]): list of vectors

    Returns:
        if_same (bool): True if the vectors have the same shape
    """
    if not vec_list: return True
    reference_shape = vec_list[0].shape
    for tensor in vec_list[1:]:
        if tensor.shape != reference_shape:
            return False
    return True

def flatten_dict(d, parent_key='', sep='@@'):
    """
    Flattens a nested dictionary into a flat dictionary.

    Parameters:
        d (dict): The nested dictionary to be flattened.
        parent_key (str): The parent key name (used for recursion).
        sep (str): The separator used to concatenate nested keys.

    Returns:
        dict: The flattened dictionary.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def unflatten_dict(d, sep='@@'):
    """
    Unflattens a flat dictionary back into a nested dictionary.

    Parameters:
        d (dict): The flat dictionary.
        sep (str): The separator used to split nested keys.

    Returns:
        dict: The restored nested dictionary.
    """
    result_dict = {}
    for key, value in d.items():
        parts = key.split(sep)
        current_level = result_dict
        for part in parts[:-1]:
            if part not in current_level:
                current_level[part] = {}
            current_level = current_level[part]
        current_level[parts[-1]] = value
    return result_dict

def dataset2sharable(dataset, batch_size=512):
    """
    Convert a dataset into sharable format, i.e., numpy arrays of features and the type information for recovering them

    Args:
        dataset (torch.utils.data.Dataset): dataset to be converted

    Returns:
        sharable_data (list[numpy.ndarray]): the numpy arrays of the dataset
    """
    first_item = dataset[0]
    if isinstance(first_item, dict):
        flattened_item = flatten_dict(first_item)
        etypes = ["$$".join(['dict']+list(flattened_item.keys()))] + [type(ei).__name__ if type(ei) not in TYPE_CANDIDATES else 'unknown' for ei in flattened_item.values()]
        def collate_func_dict(batch):
            flattened_batch = [flatten_dict(di) for di in batch]
            flattened_batch = [list(di.values()) for di in flattened_batch]
            return [list(xi) for xi in list(zip(*flattened_batch))]
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_func_dict)
        res  = [np.empty((0,))] + list(map(lambda x: list(chain(*x)), zip(*data_loader)))
        item_size = len(res)
    else:
        item_size = len(first_item)
        if not isinstance(first_item, tuple): first_item = tuple(first_item)
        etypes = [type(ei).__name__ if type(ei) not in TYPE_CANDIDATES else 'unknown' for ei in first_item]
        def collate_func(batch):
            return [list(xi) for xi in list(zip(*batch))]
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, collate_fn=collate_func)
        res = list(map(lambda x: list(chain(*x)), zip(*data_loader)))
    for j in range(item_size):
        if etypes[j] in ['int', 'float', 'str', 'int64', 'float64']:
            res[j] = np.array(res[j])
        elif etypes[j] == 'ndarray':
            if _check_vector_shapes(res[j]):
                res[j] = np.stack(res[j])
            else:
                shapes = [rjk.shape for rjk in res[j]]
                etypes[j] = {'etype': etypes[j], 'shape': shapes}
                res[j] = np.concatenate([rjk.reshape((-1, )) for rjk in res[j]])
        elif etypes[j] == 'Tensor':
            if _check_vector_shapes(res[j]):
                res[j] = torch.stack(res[j]).numpy()
            else:
                shapes = [tuple(rjk.shape) for rjk in res[j]]
                etypes[j] = {'etype': etypes[j], 'shape': shapes}
                res[j] = torch.cat([rjk.view(-1) for rjk in res[j]]).numpy()
        elif etypes[j] in ['list', 'tuple']:
            try:
                new_resj = [np.array(rjk) for rjk in res[j]]
                if _check_vector_shapes(new_resj):
                    new_resj = np.stack(new_resj)
                else:
                    shapes = [rjk.shape for rjk in new_resj]
                    etypes[j] = {'etype': etypes[j], 'shape': shapes}
                    new_resj = np.concatenate([rjk.reshape((-1, )) for rjk in new_resj])
                res[j] = new_resj
            except:
                res[j] = np.frombuffer(pickle.dumps(res[j]), dtype=np.uint8)
                etypes[j] = 'pickle'
        elif etypes[j].startswith('dict') and "@@" in etypes[j]:
            continue
        else:
            res[j] = np.frombuffer(pickle.dumps(res[j]), dtype=np.uint8)
            etypes[j] = 'pickle'
    etypes = np.frombuffer(pickle.dumps(etypes), dtype=np.uint8)
    res.append(etypes)
    return res

def sharable2dataset(sharable_data):
    """
    Recover the original data from sharable format

    Args:
        sharable_data (list[numpy.ndarray]): the numpy arrays of the dataset

    Returns:
        dataset (torch.utils.data.Dataset): the original dataset
    """

    types = pickle.loads(sharable_data.pop(-1).tobytes())
    data = []
    if types[0].startswith('dict') and "$$" in types[0]:
        all_keys = types[0].split("$$")[1:]
        types = types[1:]
        sharable_data = sharable_data[1:]
    else:
        all_keys = None
    for i in range(len(types)):
        if isinstance(types[i], dict):
            etype = types[i]['etype']
            shapes = types[i]['shape']
            offsets = [0] + np.cumsum([np.prod(s) for s in shapes]).tolist()
            tmp_data = [sharable_data[i][offsets[k]:offsets[k + 1]].reshape(shapes[k]) for k in range(len(shapes))]
            if etype == 'Tensor':
                tmp_data = [torch.from_numpy(tdi) for tdi in tmp_data]
            elif etype == 'list':
                tmp_data = [tdi.tolist() for tdi in tmp_data]
            elif etype == 'tuple':
                tmp_data = [tuple(tdi.tolist()) for tdi in tmp_data]
            data.append(tmp_data)
        elif types[i] == 'pickle':
            data.append(pickle.loads(sharable_data[i].tobytes()))
        else:
            if types[i] == 'ndarray':
                data.append(sharable_data[i])
            elif types[i] == 'Tensor':
                data.append(torch.from_numpy(sharable_data[i]))
            elif types[i] in ['int', 'float', 'str']:
                data.append(sharable_data[i].tolist())
            elif types[i] in ['int64', 'float64']:
                data.append(sharable_data[i])
    if all_keys is not None:
        return TmpDictDataset(all_keys, data)
    else:
        return TmpDataset(data)

def create_memmap_meta_for_dataset(sharable_data, name, use_uuid=True):
    """
    Map sharable data to shared memory by np.memmap

    Args:
        sharable_data (list[numpy.ndarray]): the numpy arrays of the dataset
        name (str): the file name of the memmap
        use_uuid (bool): whether to use uuid for generating the file name

    Returns:
        shm_name (str): the file name of the memmap
        dtype (np.dtype): the data type of the memmap
    """
    dtype = np.dtype([(f'{i}', sdi.dtype, sdi.shape) for i, sdi in enumerate(sharable_data)])
    shm_name = name+f"{uuid.uuid4()}" if use_uuid else name
    shm = np.memmap(shm_name, dtype=dtype, mode='w+',shape=())
    for i in range(len(sharable_data)):
        shm[f'{i}'] = sharable_data[i]
        # np.copyto(shm[f'{i}'], sharable_data[i])
    return shm_name, dtype

def create_memmap_meta_for_task(task_data, path=''):
    """
    Map task data to shared memory by np.memmap

    Args:
        task_data (dict): the task data
        path (str): the path of the temporary dictionary to store the memmap files

    Returns:
        task_meta (dict): the meta information of the task memmap
    """
    task_meta = {}
    for party in task_data:
        task_meta[party] = {}
        for data_name in task_data[party]:
            data = task_data[party][data_name]
            if data is None: continue
            sharable_data = dataset2sharable(data)
            shm_name = "_".join([party, data_name])
            if path!='': shm_name = os.path.join(os.path.abspath(path), shm_name)
            shm_name, dtype = create_memmap_meta_for_dataset(sharable_data, shm_name)
            task_meta[party][data_name] = {
                "name": shm_name,
                "dtype": dtype,
            }
    return task_meta

def load_dataset_from_memmap_meta(name, dtype):
    """
    Load one dataset from np.memmap setting's value

    Args:
        name (str): shared_memory name
        dtype (list): list of dtypes
        etype (list(str)): the element type of items in original dataset
    Returns:
        party_data (torch.utils.data.Dataset): the recovered dataset
    """
    memmap = np.memmap(name, mode='r', dtype=dtype)
    sharable_data = [memmap[f'{i}'][0] for i in range(len(dtype))]
    return sharable2dataset(sharable_data)

def load_taskdata_from_memmap_meta(task_meta):
    """
    Load task data from np.memmap setting created by create_meta_for_task

    Args:
        task_meta (dict): task meta generated by create_meta_for_task

    Returns:
        task_data (dict): task data
    """
    task_data = {}
    for party in task_meta:
        task_data[party] = {}
        for data_name in task_meta[party]:
            task_data[party][data_name] = load_dataset_from_memmap_meta(**task_meta[party][data_name])
    return task_data

def create_task_data_npy(task, train_holdout:float=0.2, test_holdout:float=0.0, local_test:bool=False, local_test_ratio:float=0.5, seed:int=0, batch_size:int=512):
    """
    Convert task data into .npy files and store them at task/.mmap/task_config_info/*.npy

    Args:
        task (str): the path of the task
        train_holdout       (flt):  the rate of holding out the validation dataset from all the local training datasets',default=0.1
        test_holdout        (flt):  the rate of holding out the validation dataset from the testing datasets owned by the server', default=0.0
        local_test          (bool): if this term is set True and train_holdout>0, (0.5*train_holdout) of data will be set as client.test_data.default=False.
        local_test_ratio    (flt):  valid if local_test is True. The ratio of local test dataset holdout from local validation data.
        seed            (int):  seed for random initialization for data train/val/test partition', default=0
        batch_size      (int): the batch size of data loader
    Returns:
        if_success          (bool): True if the task data has been successfully converted
    """
    if not os.path.exists(task): return False
    memmap_path = os.path.join(task, '.mmap', "L{:.2f}G{:.2f}T{}R{:.2f}S{}".format(train_holdout, test_holdout, int(local_test), local_test_ratio, seed))
    if not os.path.exists(memmap_path): os.makedirs(memmap_path)
    else: raise FileExistsError(f'{memmap_path} already exists')
    task_data = fuf.load_task_data(task,  train_holdout, test_holdout, local_test, local_test_ratio, seed)
    all_keys = list(task_data.keys())
    example_data = None
    for k in all_keys:
        datak = task_data[k]
        for kk in datak.keys():
            datakk = datak[kk]
            if datakk is not None and len(datakk)>0:
                example_data = datakk[0]
                break
        if example_data is not None: break
    if resource is not None:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    else:
        soft_limit = 100
    if isinstance(example_data, dict): num_elems = len(flatten_dict(example_data))+2
    elif isinstance(example_data, tuple):num_elems = len(example_data)+1
    else:
         num_elems = 2
    num_datas = sum([len(task_data[k]) for k in task_data])
    files_per_block = max(num_datas*num_elems/max(int(0.05*soft_limit), 10), 1)
    memmap_manager = MemmapManager(memmap_path, num_elems)
    task_meta = {}
    crt_block_files = 0
    for party in tqdm(task_data, desc='Creating Task: '):
        task_meta[party] = {}
        for data_name in task_data[party]:
            data = task_data[party][data_name]
            if data is None: continue
            sharable_data = dataset2sharable(data, batch_size=batch_size)
            memmap_manager.add(sharable_data, f"{party}.{data_name}")
            crt_block_files += 1
        if crt_block_files >= files_per_block:
            memmap_manager.dump()
            crt_block_files = 0
    if not memmap_manager.is_empty():
        memmap_manager.dump()
    memmap_manager.save_meta()
    print("Task data has been successfully converted into .npy")
    return True

def load_task_data_from_npy(task, train_holdout:float=0.2, test_holdout:float=0.0, local_test=False, local_test_ratio=0.5, seed=0, create=False):
    """
    Load task data from task/.mmap/task_config_info/*.npy in memmap way

    Args:
        task (str): the path of the task
        train_holdout       (flt):  the rate of holding out the validation dataset from all the local training datasets',default=0.1
        test_holdout        (flt):  the rate of holding out the validation dataset from the testing datasets owned by the server', default=0.0
        local_test          (bool): if this term is set True and train_holdout>0, (0.5*train_holdout) of data will be set as client.test_data.default=False.
        local_test_ratio    (flt):  valid if local_test is True. The ratio of local test dataset holdout from local validation data.
        dataseed            (int):  seed for random initialization for data train/val/test partition', default=0
        create              (bool): create the npy file of the task if the target files doesn't exist

    Returns:
        task_data           (dict): the task data
    """
    if not os.path.exists(task):
        raise FileNotFoundError("Fedtask {} doesn't exist".format(task))
    memmap_path = os.path.join(task, '.mmap',"L{:.2f}G{:.2f}T{}R{:.2f}S{}".format(train_holdout, test_holdout, int(local_test), local_test_ratio, seed))
    if not os.path.exists(memmap_path):
        if create:
            create_task_data_npy(task, train_holdout, test_holdout, local_test, local_test_ratio, seed)
        else:
            raise FileNotFoundError("Task npy files were not found. Use flgo.utils.shared_memory.create_task_npy to create the target.")
    memmap_manager = MemmapManager(memmap_path)
    task_data = {}
    for dname in memmap_manager.data_meta:
        party, data_name = dname.split('.')
        if party not in task_data: task_data[party] = {}
        sharable_data = memmap_manager.get(dname)
        dataset = sharable2dataset(sharable_data)
        task_data[party][data_name] = dataset
    return task_data

if __name__=='__main__':
    def worker(task_meta):
        pid = os.getpid()
        task_data = load_taskdata_from_memmap_meta(task_meta)
        selected_party = 'server'
        dataset = task_data[selected_party]['test']
        dataloader = tud.DataLoader(dataset, batch_size=2)
        for batch_id, batch_data in enumerate(dataloader):
            x, y = batch_data
            print(f"Process {pid} Xid = {id(x.numpy())} {pid} Y = {id(y.numpy())}")
            if batch_id >= 3: break
        return True
    class TenDataset(tud.Dataset):
        def __init__(self, n: int = 5, l: int = 3, r: int = 8):
            self.n = n
            self.x = [torch.rand((torch.randint(l, r, (1,)).item())) for i in range(n)]
            self.y = torch.randint(0, 9, (n,)).long()

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.x[i], self.y[i]


    class NPDataset(tud.Dataset):
        def __init__(self, n: int = 5, l: int = 3, r: int = 8):
            self.n = n
            self.x = [torch.rand((torch.randint(l, r, (1,)).item())).numpy() for i in range(n)]
            self.y = torch.randint(0, 9, (n,)).long().numpy()

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.x[i], self.y[i]


    class DictData(tud.Dataset):
        def __init__(self, n: int = 5, l: int = 3, r: int = 8):
            self.n = n
            self.x = [{'x': torch.rand((torch.randint(l, r, (1,)).item())).numpy()} for i in range(n)]
            self.y = torch.randint(0, 9, (n,)).long().numpy()

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return self.x[i], self.y[i]

    class DictData2(tud.Dataset):
        def __init__(self, n: int = 5, l: int = 3, r: int = 8):
            self.n = n
            self.x = np.random.randn(n, 10)
            self.y = np.random.randn(n, 10)

        def __len__(self):
            return self.n

        def __getitem__(self, i):
            return {'x': self.x[i], 'y':{'data':self.y[i]}}

    NUM_WORKERS = 3
    # task = './my_task'
    DATA = NPDataset
    # task_data = fuf.load_task_data(task)
    task_data = {'server':{'test': DATA(), 'val':None}, 'Client01':{'train':DATA(), 'val':DATA(), 'test':None}}
    # DEBUG: create_meta_for_task
    task_meta = {}

    tmp_data =DictData2()
    sharable_data = dataset2sharable(tmp_data)
    dataset = sharable2dataset(sharable_data)
    print('ok')
    # for party in task_data:
    #     task_meta[party] = {}
    #     for data_name in task_data[party]:
    #         data = task_data[party][data_name]
    #         if data is None: continue
    #         sharable_data, etypes = dataset2sharable(data)
    #         new_data = sharable2dataset(sharable_data, etypes)
    #         print(f"Consistency for {party} {data_name}: {all([all(new_data[i][0] == data[i][0]) for i in range(len(data))])}")
            # shm_name_list, shm_type_list, shm_shape_list= create_meta_for_dataset(sharable_data, "_".join([party, data_name]))
            # task_meta[party][data_name] = {
            #     "name": shm_name_list,
            #     "dtype": shm_type_list,
            #     "shape": shm_shape_list,
            #     'etype': etypes,
            # }
    # task_meta = create_meta_for_task(task_data)
    # worker(task_meta)
    # pros = []
    # for i in range(NUM_WORKERS):
    #     p = multiprocessing.Process(target=worker,  args=(task_meta, ))
    #     pros.append(p)
    # for p in pros: p.start()
    # for p in pros: p.join()
