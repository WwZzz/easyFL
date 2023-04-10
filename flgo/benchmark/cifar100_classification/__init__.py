r"""The usage of CIFAR100 is similar to CIFAR10. To create federated tasks based on this benchmark,
you can try the following example:

Example 1:
```python
>>> # create a IID partitioned CIFAR100 for 100 clients
>>> config = {'benchmark':{'name':'flgo.benchmark.cifar100_classification'},'partitioner':{'name': 'IIDPartitioner','para':{'num_clients':100}}}
>>> flgo.gen_task(config, task_path='./test_cifar100')
```
After running the above codes, there will be a task dictionary './test_mnist' that represents a federated task.
This task is corresponding to a static scenario where the samples in CIFAR100 is I.I.D. allocated to 100 clients,
and is also reusable for different algorithms so that a fair comparison can be easily achieved by optimizing
the same model on the same task.

Example 2:
```python
>>> # create a non-IID partitioned CIFAR100 for 100 clients
>>> niid_config = {'benchmark':{'name':'flgo.benchmark.cifar10_classification'},'partitioner':{'name': 'DirichletPartitioner','para':{'num_clients':100, 'alpha':0.5}}}
>>> flgo.gen_task(niid_config, task_path='./test_cifar_dir0.5')
```
To create a non-IID CIFAR100, the only difference is to specify different partitioners in configurations, where
the term `IIDPartitioner` is replaced by `DirichletPartitioner`. In addition, parameters of the partitioner can
be input by `para`.
"""
import flgo.benchmark.toolkits.visualization
from flgo.benchmark.cifar100_classification.model import cnn
import flgo.benchmark.toolkits.partition

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = cnn
visualize = flgo.benchmark.toolkits.visualization.visualize_by_class