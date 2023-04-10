from flgo.benchmark.vertical_mnist_classification.model import mlp
import flgo.benchmark.toolkits.partition

default_partitioner = flgo.benchmark.toolkits.partition.VerticalSplittedPartitioner
default_partition_para = {'num_parties':3}
default_model = mlp