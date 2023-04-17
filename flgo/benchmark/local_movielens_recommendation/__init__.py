import flgo.benchmark.toolkits.partition
from .model import mf
default_model = mf
default_partitioner = flgo.benchmark.toolkits.partition.IDPartitioner
default_partition_para = {'index_func':lambda x:x[0]}
