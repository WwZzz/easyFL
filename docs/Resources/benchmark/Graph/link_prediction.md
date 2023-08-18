# Overview
| **Name**                | **Dataset**                                             | **Description**                      | **Scene**     | **Download**                                                                                           | **Remark** |
|-------------------------|---------------------------------------------------------|--------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|------------|
| citeseer_link_prediction | [Citeseer](https://www.cs.toronto.edu/~kriz/cifar.html) | [See here](#citeseer_link_prediction) | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/citeseer_link_prediction.zip) |            |
| cora_link_prediction    | [Cora](http://yann.lecun.com/exdb/mnist/)               | [See here](#cora_link_prediction)    | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/cora_link_prediction.zip)    | -          |
| pubmed_link_prediction  | [PubMed](https://www.cs.toronto.edu/~kriz/cifar.html)   | [See here](#pubmed_link_prediction)  | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/pubmed_link_prediction.zip)  |            |

# Details
## **citeseer_link_prediction**
<div id="citeseer_link_prediction"></div>

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

## **cora_link_prediction**
<div id="cora_link_prediction"></div>

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

## **pubmed_link_prediction**
<div id="pubmed_link_prediction"></div>

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


