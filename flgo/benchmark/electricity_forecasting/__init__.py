import flgo.benchmark.toolkits.partition
import flgo.benchmark.electricity_forecasting.model.GRU as gru

default_partitioner = flgo.benchmark.toolkits.partition.IIDPartitioner
default_partition_para = {'num_clients':100}
default_model = gru