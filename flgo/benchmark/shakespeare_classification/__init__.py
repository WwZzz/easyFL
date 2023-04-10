from flgo.benchmark.shakespeare_classification.model import stackedlstm
import flgo.benchmark.toolkits.partition

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = stackedlstm