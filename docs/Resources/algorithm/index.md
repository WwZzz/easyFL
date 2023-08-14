# Usage
## Use Algorithm as a Module
To use the algorithm, a general way is to create an algorithm file `algorithm_name.py` in the current working project directory, and then copy the source code into it.

For example, we take next three steps to use FedAvg as a plugin. 

1. Create a new file named 'my_fedavg.py'

2. Copy the source code of FedAvg [Here](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/fedavg.py) into `my_fedavg.py`

3. put the file into the currently working project directory and use it by
```python
import my_fedavg
import flgo

flgo.init(.., algorithm=my_fedavg, ..).run()
```
## Use Algorithm as a Class
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
#---------------codes of FedAvg----------------
import flgo
from flgo.algorithm.fedbase import BasicServer as Server
from flgo.algorithm..fedbase import BasicClient as Client
#----------------------end--------------------

class my_fedavg:
    # copy the source code here
    Server = Server
    Client = Client

# Run the algorithm
flgo.init(.., algorithm=my_fedavg, ..).run()
```