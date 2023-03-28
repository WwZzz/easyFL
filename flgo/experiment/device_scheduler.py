import copy
from abc import ABCMeta, abstractmethod
import random
import torch
import time
import pynvml

class AbstractScheduler(metaclass=ABCMeta):
    @abstractmethod
    def get_available_device(self, *args, **kwargs):
        pass

class BasicScheduler(AbstractScheduler):
    def __init__(self, devices:list, *args, **kwargs):
        self.devices = devices if devices != [] else [-1]
        self.dev_index = 0
        self.process_set = set()

    def get_available_device(self, *args, **kwargs):
        self.dev_index = (self.dev_index+1)%len(self.devices)
        return self.devices[self.dev_index]

    def set_devices(self, devices:list):
        self.devices=[-1] if devices==[] else devices
        self.dev_index = self.dev_index%len(self.devices)

    def add_process(self, pid=None):
        if pid is not None:
            self.process_set.add(pid)

    def remove_process(self, pid=None):
        if pid is not None and pid in self.process_set:
            self.process_set.remove(pid)

class RandomScheduler(BasicScheduler):
    def get_available_device(self, *args, **kwargs):
        return random.choice(self.devices)

class AutoScheduler(BasicScheduler):
    def __init__(self, devices:list, put_interval = 5, mean_memory_occupated = 1000, available_interval=5, dynamic_memory_occupated=True, dynamic_condition='mean'):
        super(AutoScheduler, self).__init__(devices)
        pynvml.nvmlInit()
        crt_time = time.time()
        self.dev_state = {
            dev:{
                'avl': True,
                'time':crt_time,
                'time_put':None,
                'handle':pynvml.nvmlDeviceGetHandleByIndex(dev),
                'total_memory':0,
                'allocated_memory':0,
                'free_memory':0,
            }
            for dev in self.devices
        }
        self.put_interval = put_interval
        self.mean_memory_occupated = mean_memory_occupated
        self.available_interval = available_interval
        self.dynamic_condition = dynamic_condition
        self.dynamic_memory_occupated = dynamic_memory_occupated

    def get_available_device(self, option, *args, **kwargs):
        for dev in self.devices:
            self.flush(dev)
        all_mems = []
        for dev in self.devices:
            dev_handle = self.dev_state[dev]['handle']
            ps = pynvml.nvmlDeviceGetComputeRunningProcesses(dev_handle)
            mems = [p.usedGpuMemory for p in ps if p.pid in self.process_set]
            all_mems.extend(mems)
        if self.dynamic_memory_occupated:
            if len(all_mems)>0:
                mem = max(all_mems) if self.dynamic_condition=='max' else sum(all_mems)/len(all_mems)
                self.mean_memory_occupated = self.byte2mb(mem)
        tmp = copy.deepcopy(self.devices)
        sorted(tmp, key=lambda x:self.dev_state[x]['free_memory'])
        for dev in tmp:
            if self.check_available(dev):
                return dev
        return None

    def byte2mb(self, size):
        return int(size/1024/1024)

    def flush(self, dev):
        if dev>=0:
            handle = self.dev_state[dev]['handle']
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            self.dev_state[dev]['total_memory'] = self.byte2mb(meminfo.total)
            self.dev_state[dev]['allocated_memory'] = self.byte2mb(meminfo.used)
            self.dev_state[dev]['free_memory'] = self.byte2mb(meminfo.free)

    def check_available(self, dev):
        if dev=='-1':return True
        crt_time = time.time()
        crt_free_memory = self.dev_state[dev]['free_memory']
        target_memory = self.mean_memory_occupated
        crt_avl = crt_free_memory>target_memory
        if crt_avl:
            if self.dev_state[dev]['avl']:
                if crt_time - self.dev_state[dev]['time']>=self.available_interval:
                    if self.dev_state[dev]['time_put'] is None or crt_time-self.dev_state[dev]['time_put']>=self.put_interval:
                        self.dev_state[dev]['time_put'] = crt_time
                        return True
        if crt_avl!=self.dev_state[dev]['avl']:
            self.dev_state[dev]['avl'] = True
            self.dev_state[dev]['time'] = crt_time
        return False
