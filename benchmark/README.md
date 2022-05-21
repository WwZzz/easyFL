# How To Convert A Traditional ML Task Into A Federated One?
When someone wants to convert a traditional ML task into a federated one, the issues below will immediately arise:
* How to partition the dataset into different subdataset? How can a dataset be partitioned in the I.I.D. or non-I.I.D. manner?
* Should the existing federated optimization algorithms (e.g. fedavg) be rewrite to suit the new coming task?
* What if the metrics vary across different ML tasks? Will existing codes still work?
* Can a partitioned setting be reused by different algorithms?
* ...

Fortunately, we've developed this module to simultaneously address these problems. The main purpose of ours is to 1) provide a data pre-processing paradigm to distribute a ML dataset to different virtual clients, 2) decouple the task-specific part from federated optimization process. Now we in turn introduce how we achieve the two objectives and finally take an example of converting to help understand the whole procedure.

<p float="left">
   <img src="https://github.com/WwZzz/myfigs/blob/master/easyfl_benchmark_od.jpg" width="1000" />
</p>

## Generating FL Task From Traditional ML Task By `TaskGenerator` and `TaskReader`
coming soon...

## Decoupling Task-Specific Calculation From Federated System by `TaskCalculator`
coming soon...
