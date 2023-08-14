# Overview
| **Name**             | **Dataset**                                | **Description**                   | **Scene**     | **Download**                                                                                                             | **Comment** |
|----------------------|--------------------------------------------|-----------------------------------|---------------|--------------------------------------------------------------------------------------------------------------------------|-------------|
| mnist_classification | [MNIST](http://yann.lecun.com/exdb/mnist/) | [See here](#mnist_classification) | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/image/classification/mnist_classification.zip) | -           |
|                      |                                            |                                   |               |                                                                                                                          |             |

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

