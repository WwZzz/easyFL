import flgo.benchmark.toolkits.partition
import flgo.benchmark.bankmarketing_classification.model.lr as lr

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = lr