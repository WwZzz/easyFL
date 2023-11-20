import flgo.benchmark.toolkits.visualization
from flgo.benchmark.stl10_classification.model import resnet18_gn
from flgo.benchmark.stl10_classification.model import cnn
from flgo.benchmark.stl10_classification.model import cnn_data_aug
import flgo.benchmark.toolkits.partition

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = cnn_data_aug
visualize = flgo.benchmark.toolkits.visualization.visualize_by_class