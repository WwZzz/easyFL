import flgo
import flgo.algorithm.fedavg
import os

task = './my_task'
if not os.path.exists(task):
    flgo.gen_task('gen_config.yml', task)

runner = flgo.init(task, {'gpu':[0], 'num_rounds':1, 'log_file':True}, flgo.algorithm.fedavg)
runner.run()
