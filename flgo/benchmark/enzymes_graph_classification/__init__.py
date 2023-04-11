from flgo.benchmark.enzymes_graph_classification.model import GIN
import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
visualize = flgo.benchmark.toolkits.visualization.visualize_by_class
default_model = GIN
