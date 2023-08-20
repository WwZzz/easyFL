This is a library for sharing the resources. You can contribute to this library by 
uploading your personal algorithms, simulators, and benchmarks. All the resources here will
be opened to the public to promote interaction among developers/researchers.

# Usages
The usages of the three kinds of resources are respectively introduced in [Usage Algorithm](./algorithm/index.md), [Usage Benchmark](./benchmark/index.md), [Usage Simulator](./simulator/index.md)

# Contribute to Resources
We welcome researchers to contribute to this open-source library to share their own studies by introducing new benchmarks, novel algorithms, and more practical simulators, as we hope this can promoto the development of FL community.
To simplify integrating different kinds of resources, we have also provided easy APIs and comprehensive [tutorials](../Tutorials/index.md).
We will remark the contributors for the submitted resources in our website.

## Submit Contributions
There are two ways to submit your contributions to this platform.

### (1) Push commits to the Github repo
- **Firstly**, clone our github repo
```shell
git clone https://github.com/WwZzz/easyFL.git
```

- **Secondly**, git add your resources in proper positions (i.e. benchmark, algorithm, or simulator) in easyFL/resources
For example, 
```
└─ resources
   ├─ algorithm  # algorithm files of .py should be placed here
   │  ├─ fedavg.py  
   │  └─ ...
   ├─ benchmark  # benchmark files of .zip should be placed here
   │  ├─ mnist_classification.zip 
   │  └─ ...
   └─ simulator  # simulator files of .py should be placed here
      ├─ ...
      └─ ...
 
```

- **Thirdly**, git commit your changes with necessary information and push it to our repo, which includes

> **algorithm**: publishment, year, scene (e.g. horizontal, vertical or others)
> 
> **benchmark**: the name of dataset, the type of data, the type of the ML task
> 
> **simulator**: the name of simulator, synthetic-based or real-world-based

### (2) Contact us through email
Send resources to me through [E-mail](../about.md). The necessacy information should also be contained.


