from benchmark.generator import MNIST_TaskGenerator
from benchmark.generator import CIFAR100_TaskGenerator
from benchmark.generator import CIFAR10_TaskGenerator
from benchmark.generator import Synthetic_TaskGenerator
from benchmark.generator import FashionMNIST_TaskGenerator
from benchmark.generator import Shakespeare_TaskGenerator

if __name__ == '__main__':
    # generating the dataset of mnist-iid of 100 clients
    mnist_gen = MNIST_TaskGenerator(dist=0, num_clients=100, beta=0)
    mnist_gen.generate()

    # # generating the dataset of mnist-niid of 100 clients
    # mnist_niid_gen = MNIST_TaskGenerator(dist=3, num_clients=100, beta=2)
    # mnist_niid_gen.generate()

    # # generating the dataset of cifar10-iid of 100 clients
    # cifar10_iid_gen = CIFAR10_TaskGenerator(dist=0, num_clients=100, beta=0)
    # cifar10_iid_gen.generate()

    # # generating the dataset of cifar10-niid of 100 clients
    # cifar10_niid_gen = CIFAR10_TaskGenerator(dist=1, num_clients=100, beta=1)
    # cifar10_niid_gen.generate()

    # # generating the dataset of fashion-mnist-noniid of 3 clients ('T-shirt', 'pullover' and 'shirt')
    # FashionMNIST_gen = FashionMNIST_TaskGenerator(dist=1, num_clients=3, beta=1, selected=[0,2,6])
    # FashionMNIST_gen.generate()

    # generating the dataset of cifar100-niid of 100 clients who only have 1 kind of labels
    # cifar100_iid_gen = CIFAR100_TaskGenerator(dist=0, num_clients=100, beta=0)
    # cifar100_iid_gen.generate()

    # shk_gen = Shakespeare_TaskGenerator(dist = 5, num_clients=31, beta = 6)
    # shk_gen.generate()

    # generating synthetic-iid dataset of 30 clients (synthetic_iid, balance)
    # synthetic_iid_gen = Synthetic_TaskGenerator(dist=0, num_clients=10, beta=(0,0))
    # synthetic_iid_gen.generate()

    # generating synthetic-iid dataset of 30 clients (synthetic_iid, imbalance)
    # synthetic_iid_gen = Synthetic_TaskGenerator(dist=6, num_clients=30, beta=(0,0))
    # synthetic_iid_gen.generate()

    # # generating synthetic-noniid dataset of 30 clients (synthetic(0,0), balance)
    # synthetic_iid_gen = Synthetic_TaskGenerator(dist=1, num_clients=30, beta=(0,0))
    # synthetic_iid_gen.generate()

