from .model import default_model
import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition

default_model = default_model
default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}