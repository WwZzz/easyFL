# Overview
| **Name**                   | **Dataset**                                                          | **Description**                         | **Scene**     | **Download**                                                                                              | **Remark**      |
|----------------------------|----------------------------------------------------------------------|-----------------------------------------|---------------|-----------------------------------------------------------------------------------------------------------|-----------------|
| coco_segmentation          | [COCO](https://cocodataset.org/#detection-2016)                      | [See here](#coco_segmentation)          | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/coco_segmentation.zip)          | (under testing) |
| oxfordiiitpet_segmentation | [OxfordIIITPet](https://www.robots.ox.ac.uk/~vgg/data/pets/)         | [See here](#oxfordiiitpet_segmentation) | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/oxfordiiitpet_segmentation.zip) |                 |
| sbdataset_segmentation     | [SBDataset](http://home.bharathh.info/pubs/codes/SBD/download.html)  | [See here](#sbdataset_segmentation)     | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/sbdataset_segmentation.zip)     |                 |
| cityspaces_segmentation    | [Cityspaces](https://www.cityscapes-dataset.com/) | [See here](#cityspaces_segmentation)    | Horizontal FL | [Click Here](https://github.com/WwZzz/easyFL/raw/FLGo/resources/benchmark/cityspaces_segmentation.zip)    |                 |


# Details

## **coco_segmentation**
<div id="coco_segmentation"></div>
coco

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| FCN_ResNet50   | -                       |             | -                  |
| UNet           | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **oxfordiiitpet_segmentation**
<div id="oxfordiiitpet_segmentation"></div>
coco

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| FCN_ResNet50   | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **sbdataset_segmentation**
<div id="sbdataset_segmentation"></div>
coco

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| FCN_ResNet50   | -                       |             | -                  |
| UNet           | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
| DiversityPartitioner |           | Partitioning according to label diversity              |
| DirichletPartitioner |           | Partitioning according to dir. distribution of labels  |

## **cityspaces_segmentation**
<div id="cityspaces_segmentation"></div>

### Usage
To use this benchmark, you need to manually download the raw data into the dictionary 'cityspaces_segmentation/cityspaces/'. The necessary file contains
`leftImg8bit_trainvaltest.zip` (11GB) and `gtFine_trainvaltest.zip` (241MB).

### model
| **Model Name** | **Non-Fed Performance** | **NumPara** | **Implementation** |
|----------------|-------------------------|-------------|--------------------|
| FCN_ResNet50   | -                       |             | -                  |

### supported partitioner
| Name                 | IsDefault | Comments                                               |
|----------------------|-----------|--------------------------------------------------------|
| IIDPartitioner       | yes       |                                                        |
