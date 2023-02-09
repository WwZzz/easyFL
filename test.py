import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer
import os

# generate federated task if task doesn't exist
task = './my_task'
if not os.path.exists(task):
    flgo.gen_task('gen_config.yml', task_path=task)

# running fedavg on the specified task
runner = flgo.init(task, fedavg, {'gpu':[0], 'num_rounds':5, 'log_file':True, 'num_steps':5})
runner.run()

# visualize the experimental result
flgo.experiment.analyzer.show('res_config.yml')