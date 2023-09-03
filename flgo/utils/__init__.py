option_desc = r"""Name,Type,Description,Default Value,Comment
num_rounds,int,number of communication rounds,20,
proportion,float,proportion of clients sampled per round,0.2,
learning_rate_decay,float,learning rate decay for the training process,0.998,effective if lr_scheduler>-1
lr_scheduler,int,type of the global learning rate scheduler,-1,effective if larger than -1
early_stop,int,stop training if there is no improvement  for no smaller than the maximum rounds,-1,effective if larger than -1
num_epochs,int,number of epochs of local training,5,
num_steps,int,number of steps of local training, -1,  dominates num_epochs if larger than 0
learning_rate,float,learning rate of local training,0.1,
batch_size,int\float,batch size of local training,64,-1 means full batch and float value  means the ratio of the full datasets
optimizer,str,to select the optimizer of local training,'sgd','sgd'|'adam'|'rmsprop'|'adagrad'
clip_grad,float,clipping gradients if the max norm of  gradients \|\|g\|\| > clip_norm > 0,0.0,effective if larger than 0.0
momentum,float,momentum of local training,0.0,
weight_decay,float,weight decay of local training,0.0,
num_edge_rounds,int,number of edge rounds in hierFL,5,effective if scene is 'hierarchical'
algo_para,int\list,algorithm-specific hyper-parameters,[],the order should be consistent with  the claim
sample,str,to select sampling form, 'uniform', 'uniform'|'md'| 'full'| x+'_with_availability'
aggregate,str,to select aggregation form, 'other', 'uniform'|'weighted_com'|'weighted_scale'|'other']
train_holdout,float,the rate of holding out the validation  dataset from all the local training datasets,0.1,
test_holdout,float,the rate of holding out the validation  dataset from the testing datasets owned by  the server,0.0,effective if the server has  no validation data
local_test,bool,the local validation data will be equally  split into validation and testing parts  if True,False,
seed,int,seed for all the random modules,0,
dataseed,int,seed for all the random modules for data train/val/test partition,0,
gpu,int\list,GPU IDs and empty input means using CPU,[],
server_with_cpu,bool,the model parameters will be stored in  the memory if True,False,
num_parallels,int,the number of parallels during communications,1,
num_workers,int,the number of workers of DataLoader,0,
pin_memory,bool,1)pin_memory of DataLoader and 2) load  data directly into memory,False,
test_batch_size,int,the batch_size used in testing phase,512,
availability,str,to select client availability mode,'IDL', 'IDL'|'YMF'|'MDF'|'LDF'|'YFF'|'HOMO'|'LN'|'SLN'|'YC'
connectivity,str,to select client connectivity mode,'IDL','IDL'|'HOMO'
completeness,str,to select client completeness mode,'IDL','IDL'|'PDU'|'FSU'|'ADU'|'ASU'
responsiveness,str,to select client responsiveness mode,'IDL','IDL'|'LN'|'UNI'
log_level,str,the level of logger,'INFO','INFO'|'DEBUG'
log_file,bool,whether log to file and default  value is False,False,
no_log_console,bool,whether log to screen and default  value is True,True,
no_overwrite,bool,whether to overwrite the old result,False,
eval_interval,int,evaluate every __ rounds;,1,
"""