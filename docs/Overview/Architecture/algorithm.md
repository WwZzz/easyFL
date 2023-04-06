# Algorithm
In FLGo, each algorithm is described by an independent file consisting of the objects 
(i.e. server and clients in horizontal FL) with their actions. 
## Horizontal FL
![algo_example](../../img/overview_flgo_algo.png)
A classical procedure of FL training process is as shown in the figure above, where the server iteratively 
broadcasts the global model to a subset of clients and aggregates the received locally 
trained models from them. Following this scheme, a great number of FL algorithms can be 
easily implemented by FLGo. For example, to implement methods that customize the local 
training process (e.g. FedProx, MOON), developers only need to modify the function 
`client.train(...)`. And a series of sampling strategies can be realized by only replacing 
the function `server.sample() `. We also provide comprehensive tutorial for using FLGo 
to implement the state of the art algorithms. In addition, asynchronous algorithms can 
share the same scheme with synchronous algorithms in FLGo, where developers only need to 
concern about the sampling strategy and how to deal with the currently received packages 
from clients at each moment. 

## Vertical FL
To be completed.