# Use Algorithm as File
To use the algorithm, a general way is to create an algorithm file `algorithm_name.py`, and then copy the source code into it.

For example, we take next three steps to use FedAvg as a plugin. 

1. Create a new file named 'my_fedavg.py'

2. Copy the source code of FedAvg [Here](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resource/algorithm/fedavg.py) into `my_fedavg.py`

3. put the file into the project file and use it like
```python
import my_fedavg
import flgo

flgo.init(.., algorithm=my_fedavg, ..)
```
# Use Algorithm by Class
Another way to use the algorithm as a plugin is to Create a class instead of a file. 

1. Copy the source code (i.e. the source code is as follows)
```python
# Example: source code of FedAvg
from flgo.algorithm.fedbase import BasicServer as Server
from flgo.algorithm..fedbase import BasicClient as Client
```

2. Create a new class like

```python
# Example: source code of FedAvg
import flgo
from flgo.algorithm.fedbase import BasicServer as Server
from flgo.algorithm..fedbase import BasicClient as Client


class my_fedavg:
    # copy the source code here
    Server = Server
    Client = Client

# 
flgo.init(.., algorithm=my_fedavg, ..)
```