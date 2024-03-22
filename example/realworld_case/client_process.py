import flgo
import flgo.algorithm.realbase as realbase

task = 'real_synthetic'
target_path = 'client_1'
flgo.pull_task_from_('tcp://127.0.0.1:7777', task, target_path=target_path)
flgo.init(task, realbase, scene='real_hclient').run(server_ip='127.0.0.1', server_port="5555")
