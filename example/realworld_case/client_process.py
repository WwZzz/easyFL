import flgo
import flgo.algorithm.realbase as realbase

task = 'real_synthetic'
flgo.pull_task_from_('tcp://10.24.116.58:7777', task)
flgo.init(task, realbase, scene='real_hclient').run(server_ip='10.24.116.58', server_port="5555")
