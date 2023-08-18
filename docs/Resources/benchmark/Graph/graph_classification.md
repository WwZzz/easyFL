# Overview
| **Name**                | **Dataset**                                                 | **Description**                      | **Scene**     | **Download**                                                                                           | **Remark** |
|-------------------------|-------------------------------------------------------------|--------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|------------|
| enzymes_graph_classification    | [ENZYMES]()                                                 | [See here](#enzymes_graph_classification)    | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/enzymes_graph_classification.zip)    | -          |
| mutag_graph_classification  | [MUTAG]()        | [See here](#mutag_graph_classification)  | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/mutag_graph_classification.zip)  |            |

# Details

## **enzymes_graph_classification**
<div id="enzymes_graph_classification"></div>

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| GIN            | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **mutag_graph_classification**
<div id="mutag_graph_classification"></div>

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| GCN            | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |
