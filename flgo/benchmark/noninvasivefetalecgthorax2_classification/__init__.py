import flgo.benchmark.toolkits.partition
import flgo.benchmark.noninvasivefetalecgthorax2_classification.model.cnn as cnn

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = cnn