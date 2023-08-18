# Overview
| **Name**                | **Dataset**                                         | **Description**                      | **Scene**     | **Download**                                                                                           | **Remark** |
|-------------------------|-----------------------------------------------------|--------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|------------|
| multi30k_translation    | [Multi30k]()       | [See here](#multi30k_translation)    | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/multi30k_translation.zip)    | -          |

# Details

## **multi30k_translation**
<div id="multi30k_translation"></div>

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| Transformer    | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |
