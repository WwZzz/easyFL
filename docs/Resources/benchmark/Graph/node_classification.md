# Overview
| **Name**                | **Dataset**                                             | **Description**                      | **Scene**     | **Download**                                                                                           | **Remark** |
|-------------------------|---------------------------------------------------------|--------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|------------|
| citeseer_node_classification | [Citeseer](https://www.cs.toronto.edu/~kriz/cifar.html) | [See here](#citeseer_node_classification) | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/citeseer_node_classification.zip) |            |
| cora_node_classification    | [Cora](http://yann.lecun.com/exdb/mnist/)               | [See here](#cora_node_classification)    | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/cora_node_classification.zip)    | -          |
| pubmed_node_classification  | [PubMed](https://www.cs.toronto.edu/~kriz/cifar.html)   | [See here](#pubmed_node_classification)  | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/pubmed_node_classification.zip)  |            |

# Details
## **citeseer_node_classification**
<div id="citeseer_node_classification"></div>
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

## **cora_node_classification**
<div id="cora_node_classification"></div>
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

## **pubmed_node_classification**
<div id="pubmed_node_classification"></div>
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


