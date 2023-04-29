from .utils.fflow import init, gen_task, gen_task_from_para, gen_benchmark_from_file,tune, run_in_parallel, multi_init_and_run

communicator = None

class VirtualCommunicator:
    """
    Communicator that simulates the communication phase between any two objects
    """
    def __init__(self, objects):
        self.objects_map = {obj.id:obj for obj in objects}
        self.objects = objects

    def request(self, source, target, package):
        # send package to the target object with `package` and `mtype`, and then listen from it
        return self.objects_map[target].message_handler(package)
