# Logger and analyzer
![tracer](https://raw.githubusercontent.com/WwZzz/myfigs/master/overview_exp.png)
Although there are already several comprehensive experiment managers (e.g. wandb, 
tensorboard), our `Experiment` module is compatible with them and enable 
customizing experiments in a non-intrusive way to the codes, where users can create a 
`logger` by modifying some APIs to track variables of interest and specify the customized 
`logger` in optional parameters. 

After the `logger` stores the running-time information into records, the `analyzer` can read 
them from the disk. A filter is designed to enable only selecting records of interest, and 
several APIs are provided for quickly visualizing and analyzing the results by few codes.

# Device Scheduler
To be complete.