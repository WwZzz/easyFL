import flgo
import flgo.algorithm.realbase as realbase

server_ip = '127.0.0.1'

flgo.pull_task_from_('tcp://%s:7777'%server_ip, 'real_synthetic')
flgo.init('real_synthetic', realbase, scene='real_hclient').run(server_ip=server_ip, server_port="5555")
