The results are stored in ../task/dataset_name/record/ and named using the format below:
  ```'methodName_'+'hyper-paras{}_'+'num_rounds{}_'+'batchsize{}_'+'localEpochs{}_'+'lr{}_'+'proportion{}_'+'seed{}'+'.json'```.
For example, ```fedfv_alpha0.334_tau0_r200_b-1_e1_lr0.01_p1.0_seed0``` means using method FedFV to run the selected task with the dict of paras:
```
{
  alpha:          0.334,
  tau:            0,
  num_rounds:     200,
  batch_size:     -1, //full batch
  epoch:          1,
  learning_rate:  0.01,
  proportion:     1
  seed:           0
}
```
To get the tables and figures in the paper, one could run the corresponding commands according to the names of records in ./task/dataset_name/record.
