r"""
Synthetic dataset is from the article 'Federated Optimization in Heterogeneous Networks' (http://arxiv.org/abs/1812.06127) of Li et. al.
The para of this benchmark is
Args:::
    alpha (float): Synthetic(alpha, beta), default value is 0
    beta (float): Synthetic(alpha, beta), default value is 0
    num_clients (int): the number of all the clients, default value is 30
    imbalance (float): the degree of data-size-imbalance ($imbalance \in [0,1]$), default value is 0.0
    mean_datavol (int): the mean data size of all the clients, default value is 400
    dimension (int): the dimension of the feature, default value is 60
    num_classes (int): the number of classes in the label set, default value is 10

Example::
```python
    >>> import flgo
    >>> import flgo.benchmark.synthetic_regression as synthetic
    >>> config = {'benchmark':{'name':synthetic, 'para':{'alpha':1., 'beta':1., 'num_clients':30}}}
    >>> flgo.gen_task(config, './my_synthetic11')
```
"""
from flgo.benchmark.synthetic_regression.model import lr

default_model = lr