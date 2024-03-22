import flgo
import flgo.benchmark.synthetic_regression as synthetic
import flgo.algorithm.realbase as realbase

server_ip = None

task = './real_synthetic'
flgo.gen_real_task(synthetic, 'synthetic_dataset.py', task)
flgo.init(task, realbase, {'save_checkpoint':'1', 'proportion':1.0}, scene='real_hserver').run(ip=server_ip, port="5555", port_task='7777')