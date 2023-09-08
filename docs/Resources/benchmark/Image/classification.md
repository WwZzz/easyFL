# Overview
| **Name**                 | **Dataset**                                                 | **Description**                       | **Scene**      | **Download**                                                                                            | **Remark**   |
|--------------------------|-------------------------------------------------------------|---------------------------------------|----------------|---------------------------------------------------------------------------------------------------------|--------------|
| mnist_classification     | [MNIST](http://yann.lecun.com/exdb/mnist/)                  | [See here](#mnist_classification)     | Horizontal FL  | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/mnist_classification.zip)     | -            |
| cifar10_classification   | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)      | [See here](#cifar10_classification)   | Horizontal FL  | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/cifar10_classification.zip)   |              |
| cifar100_classification  | [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)     | [See here](#cifar100_classification)  | Horizontal FL  | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/cifar100_classification.zip)  |              |
| svhn_classification      | [SVHN](http://ufldl.stanford.edu/housenumbers/)             | [See here](#svhn_classification)      | Horizontal FL  | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/svhn_classification.zip)      | -            |
| fashion_classification   | [FASHION](https://github.com/zalandoresearch/fashion-mnist) | [See here](#fashion_classification)   | Horizontal FL  | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/fashion_classification.zip)   | -            |
| domainnet_classification | [DomainNet](https://ai.bu.edu/M3SDA)                        | [See here](#domainnet_classification) | Horizontal FL  | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/domainnet_classification.zip) | Feature Skew |
|                          |                                                             |                                       |                |                                                                                                         |              |

# Details

## **mnist_classification**
<div id="mnist_classification"></div>

![mnist](https://production-media.paperswithcode.com/datasets/MNIST-0000000001-2e09631a_09liOmx.jpg)

Federated MNIST classification is a commonly used benchmark in FL. It assumes different virtual clients having non-overlapping samples from MNIST dataset.

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| cnn            | -                       |             | -                  |
| mlp            | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **cifar10_classification**
<div id="cifar10_classification"></div>

![cifar10](https://production-media.paperswithcode.com/datasets/4fdf2b82-2bc3-4f97-ba51-400322b228b1.png)

Federated CIFAR10 classification is a commonly used benchmark in FL. It assumes different virtual clients having non-overlapping samples from CIFAR10 dataset.

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| cnn            | -                       |             | -                  |
| mlp            | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **cifar100_classification**
<div id="cifar100_classification"></div>
Federated CIFAR100 classification is a commonly used benchmark in FL. It assumes different virtual clients having non-overlapping samples from CIFAR100 dataset.

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| cnn            | -                       |             | -                  |
| mlp            | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **svhn_classification**
<div id="svhn_classification"></div>
Federated SVHN classification is a commonly used benchmark in FL. It assumes different virtual clients having non-overlapping samples from SVHN dataset.

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| cnn            | -                       |             | -                  |
| mlp            | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **fashion_classification**
<div id="fashion_classification"></div>
Federated Fashion classification is a commonly used benchmark in FL. It assumes different virtual clients having non-overlapping samples from FashionMNIST dataset.

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| lr             | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **domainnet_classification**
<div id="domainnet_classification"></div>

![domainnet](http://ai.bu.edu/M3SDA/imgs/data_examples.png)

DomainNet contains images of the same labels but different styles (i.e. 6 styles), which can be used to investigate the influence of feature skew in FL.
The paper is available at [link](https://arxiv.org/pdf/1812.01754.pdf)

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| AlexNet        | -                       |             | -                  |
| resnet18       |                         |             |                    |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |


