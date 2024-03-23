import flgo
import synthetic_regression as synthetic
import flgo.algorithm.realbase as realbase

server_ip = '127.0.0.1'

task = './real_synthetic'
flgo.gen_real_task(synthetic, 'synthetic_dataset.py', task)
server_runner = flgo.init(task, realbase, {'save_checkpoint':'1', 'proportion':1.0, 'num_rounds':30, 'learning_rate':0.1, 'num_epochs':5}, scene='real_hserver')
server_runner.run(ip=server_ip, port="5555", port_task='7777')