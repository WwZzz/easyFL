import flgo.algorithm.fedavg as fedavg
import flgo.experiment.analyzer
import fedprox

task = './my_task'
runner_fedavg = flgo.init(task, fedavg, {'gpu':[0], 'num_rounds':5, 'log_file':True, 'num_steps':5})
runner_fedavg.run()

runner_fedprox = flgo.init(task, fedprox, {'gpu':[0], 'num_rounds':5, 'log_file':True, 'num_steps':5})
runner_fedprox.run()

flgo.experiment.analyzer.show('res_fedprox.yml')

