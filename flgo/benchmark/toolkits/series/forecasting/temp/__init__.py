from .model import default_model
import flgo.benchmark.partition as fbp

default_model = default_model
default_partitioner = fbp.DeconcatPartitioner
default_partition_para = {}

