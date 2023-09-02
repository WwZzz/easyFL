# Personalized FL

To use these algorithms, The term `Logger` should be set as `flgo.experiment.logger.pfl_logger.PFLLogger`. For example,
```python
import flgo
from flgo.experiment.logger.pfl_logger import PFLLogger
task = './my_task'
# Download codes of ditto and copy it into file 'ditto.py'
import ditto
runner = flgo.init(task, ditto, {'gpu':[0,],'log_file':True, 'num_steps':5}, Logger=PFLLogger)
runner.run()
```

| **Name**   | **Download**                                                                                        | **Publish**        | **Paper Link**                                                | **Remark** |
|------------|-----------------------------------------------------------------------------------------------------|--------------------|---------------------------------------------------------------|------------|
| Ditto      | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/ditto.py)     | ICML 2021          | [Click](http://arxiv.org/abs/2007.14390)                      |            |
| FedALA     | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/fedala.py)    | AAAI 2023          | [Click](http://arxiv.org/abs/2212.01197)                      |            |
| FedRep     | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/fedrep.py)    | ICML 2021          | [Click](http://arxiv.org/abs/2102.07078)                      |            |
| pFedMe     | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/pfedme.py)    | NIPS 2020          | [Click](http://arxiv.org/abs/2006.08848)                      |            |                                         |
| Per-FedAvg | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/perfedavg.py) | NIPS 2020          | [Click](http://arxiv.org/abs/2002.07948)                      |            |
| FedAMP     | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/fedamp.py)    | AAAI 2021          | [Click](http://arxiv.org/abs/2007.03797)                      |            |
| FedFomo    | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/fedfomo.py)   | ICLR 2021          | [Click](http://arxiv.org/abs/2012.08565)                      |            |
| LG-FedAvg  | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/lgfedavg.py)  | NIPS 2019 workshop | [Click](http://arxiv.org/abs/2001.01523)                      |            |
| pFedHN     | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/pfedhn.py)    | ICML 2021          | [Click](https://proceedings.mlr.press/v139/shamsian21a.html)  |            |
| Fed-ROD    | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/fedrod.py)    | ICLR 2023          | [Click](https://openreview.net/forum?id=I1hQbx10Kxn)          |            |
| FedPAC     | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/fedpac.py)    | ICLR 2023          | [Click](http://arxiv.org/abs/2306.11867)                                                              |            |
| FedPer     | [source code](https://raw.githubusercontent.com/WwZzz/easyFL/FLGo/resources/algorithm/fedper.py)    | AISTATS 2020       |                                                               |            |
|            |                                                                                                     |                    |                                                               |            |
