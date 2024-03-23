import flgo
import synthetic_regression as synthetic
import flgo.algorithm.realbase as realbase

server_ip = '127.0.0.1'

task = './real_synthetic'
flgo.gen_real_task(synthetic, 'synthetic_dataset.py', task)
server_runner = flgo.init(task, realbase, {'save_checkpoint':'1', 'proportion':1.0, 'num_rounds':30, 'learning_rate':0.1, 'num_epochs':5}, scene='real_hserver')
# # clients who lose connections for >=availability_timeout seconds will be viewed unavailable. Default value is 30
# server_runner.set_availability_timeout(30)
# # clients who doesn't send back model till communication_timeout seconds will be viewed delayed. Default value is 30
# server_runner.set_communication_timeout(10)
# # wait for training start until num_clients>=2. Default condition is the same
# server_runner.register_start_condition(lambda server: server.num_clients>=2)
server_runner.run(ip=server_ip, port="5555", port_task='7777')