# Federated Learning with Fair Averaging 
This repository is an implementation of the fair federated algorithm discussed in
```
    Zheng Wang, Xiaoliang Fan, Jianzhong Qi, Chenglu Wen, Cheng Wang and Rongshan Yu
    Federated Learning with Fair Averaging (ijcai-2021), 
    Apr. 30, 2021. 
    https://arxiv.org/abs/2104.14937
```
And everyone can use the experimental platform to quickly realize and compare popular centralized federated learning algorithms.

## Abstract
 Fairness has emerged as a critical problem in federated learning (FL). In this work, we identify a cause of unfairness in FL -- *conflicting* gradients with large differences in the magnitudes. To address this issue, we propose the federated fair averaging (FedFV) algorithm to mitigate potential conflicts among clients before averaging their gradients. We first use the cosine similarity to detect gradient conflicts, and then iteratively eliminate such conflicts by modifying both the direction and the magnitude of the gradients. We further show the theoretical foundation of FedFV to mitigate the issue conflicting gradients and converge to Pareto stationary solutions. Extensive  experiments on a suite of federated datasets confirm that FedFV compares favorably against state-of-the-art methods in terms of fairness, accuracy and efficiency.


## Setup
Requirements:
```
pytorch=1.3.1
torchvision=0.4.2
cvxopt=1.2.0 (is required for fedmgda+)
```
## Quick Start
### Options

### task
The 
