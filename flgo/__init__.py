from .utils.fflow import init
from .utils.fflow import gen_task

communicator = None

class VirtualCommunicator:
    def __init__(self, objects):
        self.objects = objects

    def request(self, source, target, package, mtype):
        # send package to the target object with `package` and `mtype`, and then listen from it
        return self.objects[target+1].message_handler(package, mtype)
