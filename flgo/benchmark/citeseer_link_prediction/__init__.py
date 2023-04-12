from flgo.benchmark.citeseer_link_prediction.model import GCN
import flgo.benchmark.toolkits.visualization
import flgo.benchmark.toolkits.partition
default_partitioner = flgo.benchmark.toolkits.partition.NodeLouvainPartitioner
default_partition_para= {'num_clients':10}
visualize = flgo.benchmark.toolkits.visualization.visualize_by_community
default_model = GCN