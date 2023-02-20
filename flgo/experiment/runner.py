import pynvml
import time
import queue
import sys
import os
import subprocess
import prettytable as pt
import argparse
import yaml
import itertools

pynvml.nvmlInit()
NUM_DEVICES = pynvml.nvmlDeviceGetCount()
# The maximum of the allocated memory(MB) of a gpu to a command
LEAST_FREE_MEMORY = 100
# run commands if device is being available for more than AVAILABLE_INTERVAL (seconds)
AVAILABLE_PERIOD = 30
# check interval
CHECK_INTERVAL = 1
# root path for cmd
COMMAND_WORKING_DIR = os.path.abspath('..')
# root path for bin
BIN_DIR = os.path.dirname(sys.executable)

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    # basic settings
    parser.add_argument('--m', help='the least memory needed for each command', type=int, default=1500)
    parser.add_argument('--p',  help='the period of confirming the GPU is being adequately free (measured by seconds), default as 30s', type=int, default=5)
    try:
        option = vars(parser.parse_args())
    except IOError as msg:
        parser.error(str(msg))
    global LEAST_FREE_MEMORY, AVAILABLE_PERIOD, CHECK_INTERVAL, BIN_DIR
    # The maximum of the allocated memory(MB) of a gpu to a command
    LEAST_FREE_MEMORY = option['m']
    AVAILABLE_PERIOD = option['p']
    if BIN_DIR[-1] != '/': BIN_DIR += '/'
    with open(option['config']) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def config2cmd(cfg):
    cmds = []
    list_kv = [(k,v) for k,v in cfg.items() if k!='gpu' and type(v) is list]
    list_k = [e[0] for e in list_kv]
    list_v = (e[1] for e in list_kv)
    combination = list(itertools.product(*list_v))
    cmd_components = ['python', 'main.py']
    common_part= ['--'+k+' '+str(v) for k,v in cfg.items() if k!='gpu' and k not in list_k]
    cmd_components.extend(common_part)
    cmd_head = ' '.join(cmd_components)
    for c in combination:
        kv_pair = ['--'+k+' '+str(v) for k,v in zip([key for key in list_k], c)]
        cmd = cmd_head + ' ' + ' '.join(kv_pair)
        cmds.append(cmd)
    # 要改####################################
    if 'gpu' in cfg.keys():
        devices = cfg['gpu']
        if type(devices) is not list: devices = [devices]
        crt_dev_idx = 0
        for ci in range(len(cmds)):
            cmds[ci] = cmds[ci]+' --gpu '+ str(devices[crt_dev_idx])
            crt_dev_idx = (crt_dev_idx+1)%len(devices)
    return cmds

def clear_screen():
    # for windows
    if os.name == 'nt':
        _ = os.system('cls')

    # for mac and linux(here, os.name is 'posix')
    else:
        _ = os.system('clear')


def output_table(dev_list):
    create_time = time.asctime(time.localtime(time.time()))
    tb = pt.PrettyTable()
    tb.title = "Create Time: " + create_time
    tb.field_names = ['Device', 'Finished\\Total', 'Running', "Memory", "Power"]
    clear_screen()
    tb.clear_rows()
    for dm in dev_list:
        if dm.num_commands > 0:
            cmd_state = dm.get_command_state()
            dev_state = dm.get_device_state()
            tb.add_row([dm.name,
                        "{}\\{}".format(cmd_state['Finished'], cmd_state['Total']),
                        cmd_state['Running'],
                        "{}\\{} MB".format(dev_state['allocated'], dev_state['total']),
                        "{}\\{} W".format(dev_state['power_usage'], dev_state['power_limit'])
                        ])
    print(tb)

class DeviceManager:
    def __init__(self, id=-1):
        self.id = id
        self.condition_satisfied = False
        self.cmd_queue = queue.Queue(maxsize=1000)
        self.name = "cuda:{}".format(id) if id > -1 else "cpu"
        self.num_processing = 0
        self.create_time = time.time()
        self.ready_start = time.time()
        self.ready_end = self.ready_start
        self.ready_interval = self.ready_end - self.ready_start
        self.num_finished = 0
        self.num_commands = 0
        self.error_num = 0
        self.pool = set()
        self.error_commands = []
        if id == -1:
            self.using_gpu = False
            self.handle = None
            self.total_memory = 9999999999
            self.free_memory = 9999999999
            self.allocated_memory = 0
            self.power_limit = 99999
            self.power_usage = 0
        else:
            self.using_gpu = True
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(id)
            self.flush_device_info()
        self.max_num_processing = self.total_memory / LEAST_FREE_MEMORY

    def flush(self):
        if self.check_dev_ready() and not self.queue_empty():
            self.run_cmd()
        # check error and finished
        _ = [self.put_cmd(p.args) for p in self.pool if p.poll() is not None and p.poll() != 0]
        self.num_finished += len([p for p in self.pool if p.poll() == 0])
        self.pool.difference_update([p for p in self.pool if p.poll() is not None])
        self.num_processing = len(self.pool)

    def flush_device_info(self):
        if self.using_gpu:
            self.meminfo = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            self.total_memory = self.byte2mb(self.meminfo.total)
            self.allocated_memory = self.byte2mb(self.meminfo.used)
            self.free_memory = self.byte2mb(self.meminfo.free)
            self.power_limit = pynvml.nvmlDeviceGetEnforcedPowerLimit(self.handle) / 1000
            self.power_usage = pynvml.nvmlDeviceGetPowerUsage(self.handle) / 1000

    def byte2mb(self, size):
        return int(size / 1024 / 1024)

    def get_device_state(self):
        self.flush_device_info()
        return {
            'name': self.name,
            'total': self.total_memory,
            'allocated': self.allocated_memory,
            'free': self.free_memory,
            'power_limit': self.power_limit,
            'power_usage': self.power_usage
        }

    def get_command_state(self):
        return {"Running": self.num_processing, "Total": self.num_commands, "Finished": self.num_finished}

    def condition(self):
        crt_state = self.get_device_state()
        return self.num_processing < self.max_num_processing and crt_state['free'] > LEAST_FREE_MEMORY

    def check_dev_ready(self):
        if self.condition():
            self.ready_end = time.time()
            self.ready_interval = self.ready_end - self.ready_start
        else:
            self.ready_start = time.time()
            self.ready_interval = self.ready_end - self.ready_start
        if self.ready_interval > AVAILABLE_PERIOD:
            return True
        else:
            return False

    def put_cmd(self, cmd):
        if cmd == "": return
        self.cmd_queue.put(cmd)

    def get_cmd(self):
        if self.queue_empty():
            return ""
        return self.cmd_queue.get()

    def queue_empty(self):
        return self.cmd_queue.empty()

    def run_cmd(self):
        cmd = self.get_cmd()
        # 要改####################################
        if cmd != "":
            print("Running cmd: " + cmd)
            self.ready_start = time.time()
            p = subprocess.Popen(cmd, shell=True, cwd=COMMAND_WORKING_DIR, stdout=subprocess.PIPE,
                                 stdin=subprocess.PIPE)
            self.pool.add(p)

# init dev
dev_list = [DeviceManager(i) for i in range(NUM_DEVICES)]
dev_list.append(DeviceManager(-1))
config = read_option()
cmds = config2cmd(config)
num_cmds = len(cmds)
# initialize queues of devices with cmds
num_effective_cmds = 0

# 要改####################################
for cmd in cmds:
    if cmd[-1] == '\n':
        cmd = cmd[:-1]
    cmd = cmd.strip()
    if cmd == "":
        continue
    num_effective_cmds += 1
    cmd = BIN_DIR + cmd
    # choose device
    idx = cmd.find('gpu')
    if idx == -1:
        dev_list[-1].put_cmd(cmd)
    else:
        id = int(cmd[idx + 4])
        dev_list[id].put_cmd(cmd)

for dev in dev_list:
    dev.num_commands = dev.cmd_queue.qsize()

while True:
    num_finished = sum([dm.num_finished for dm in dev_list])
    output_table(dev_list)
    for dm in dev_list:
        dm.flush()
    # 要改####################################
    num_running = sum([d.num_processing for d in dev_list])
    num_unfinished = sum([d.cmd_queue.qsize() for d in dev_list])
    if num_finished >= num_effective_cmds or (num_running == 0 and num_unfinished == 0):
        output_table(dev_list)
        break
    time.sleep(CHECK_INTERVAL)
