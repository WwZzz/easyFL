# Overview
| **Name**                | **Dataset**                                         | **Description**                      | **Scene**     | **Download**                                                                                           | **Remark** |
|-------------------------|-----------------------------------------------------|--------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|------------|
| agnews_classification    | [AGNEWS](http://yann.lecun.com/exdb/mnist/)         | [See here](#agnews_classification)    | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/agnews_classification.zip)    | -          |
| imdb_classification  | [IMDB](https://www.cs.toronto.edu/~kriz/cifar.html) | [See here](#imdb_classification)  | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/imdb_classification.zip)  |            |
| sst2_classification | [SST2](https://www.cs.toronto.edu/~kriz/cifar.html) | [See here](#sst2_classification) | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/sst2_classification.zip) |            |

# Details

## **agnews_classification**
<div id="agnews_classification"></div>

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| TextClassificationModel            | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **imdb_classification**
<div id="imdb_classification"></div>
-
### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| TextClassificationModel            | -                       |             | -                  |



### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **sst2_classification**
<div id="sst2_classification"></div>
-
### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| TextClassificationModel            | -                       |             | -                  |


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

