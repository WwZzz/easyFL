import flgo
import flgo.algorithm.fedavg
import flgo.experiment.analyzer
import os
#
# task = './my_task'
# if not os.path.exists(task):
#     flgo.gen_task('gen_config.yml', task)
#
# runner = flgo.init(task, {'gpu':[0], 'num_rounds':1, 'log_file':True}, flgo.algorithm.fedavg)
# runner.run()

flgo.experiment.analyzer.show('res_config.yml')