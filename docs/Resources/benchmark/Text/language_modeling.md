# Overview
| **Name**                | **Dataset**                                              | **Description**                      | **Scene**     | **Download**                                                                                           | **Remark** |
|-------------------------|----------------------------------------------------------|--------------------------------------|---------------|--------------------------------------------------------------------------------------------------------|------------|
| penntreebank_modeling    | [PennTreebank]()                                         | [See here](#penntreebank_modeling)    | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/penntreebank_modeling.zip)    | -          |
| wikitext2_modeling | [WikiText2]() | [See here](#wikitext2_modeling) | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/wikitext2_modeling.zip) |            |

# Details

## **penntreebank_modeling**
<div id="penntreebank_modeling"></div>

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


## **wikitext2_modeling**
<div id="wikitext2_modeling"></div>
-
### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| Transformer            | -                       |             | -                  |


### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

