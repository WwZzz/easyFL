# Overview
| **Name**                | **Dataset**                                                             | **Description**                      | **Scene**     | **Download**                                                                                           | **Comment** |
|-------------------------|-------------------------------------------------------------------------|--------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|-------------|
| mnist_classification    | [MNIST](http://yann.lecun.com/exdb/mnist/)                              | [See here](#mnist_classification)    | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/mnist_classification.zip)    | -           |
| cifar10_classification  | [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)                  | [See here](#cifar10_classification)  | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/cifar10_classification.zip)  |             |
| cifar100_classification | [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)                 | [See here](#cifar100_classification) | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/cifar100_classification.zip) |             |
| svhn_classification     | [SVHN](http://ufldl.stanford.edu/housenumbers/)                         | [See here](#svhn_classification)     | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/svhn_classification.zip)     | -           |
| fashion_classification  | [FASHION](https://github.com/zalandoresearch/fashion-mnist)             | [See here](#fashion_classification)  | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/fashion_classification.zip)  | -           |

# Details

## **mnist_classification**
<div id="mnist_classification"></div>
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

[//]: # (## **bmk_name**)

[//]: # (<div id="bmk_name"></div>)

[//]: # ()
[//]: # (description here.)

[//]: # ()
[//]: # (### model)

[//]: # (| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |)

[//]: # (|----------------|-------------------------|-------------|--------------------|)

[//]: # (| -              | -                       |             | -                  |)

[//]: # ()
[//]: # (### supported partitioner)

[//]: # (| Name                 | IsDefault | Comments                                               |)

[//]: # (|----------------------|-----------|--------------------------------------------------------|)

[//]: # (| IIDPartitioner       | yes       |                                                        |)

[//]: # (| DiversityPartitioner |           | Partitioning according to label diversity              |)

[//]: # (| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |)

