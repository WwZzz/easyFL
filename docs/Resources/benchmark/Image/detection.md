# Overview
| **Name**                | **Dataset**                                                 | **Description**                      | **Scene**     | **Download**                                                                                          | **Remark**      |
|-------------------------|-------------------------------------------------------------|--------------------------------------|---------------|-------------------------------------------------------------------------------------------------------|-----------------|
| coco_detection          | [COCO](https://cocodataset.org/#detection-2016)                   | [See here](#coco_detection)     | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/coco_detection.zip)         | (under testing) |
| voc_detection           | [VOC](http://host.robots.ox.ac.uk/pascal/VOC/)          | [See here](#voc_detection)      | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/voc_detection.zip)  |                 |

# Details

## **coco_detection**
<div id="coco_detection"></div>
coco

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| FasterRCNN     | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **voc_detection**
<div id="voc_detection"></div>
coco

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| FasterRCNN     | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |