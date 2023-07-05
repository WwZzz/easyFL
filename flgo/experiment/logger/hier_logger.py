from flgo.experiment.logger import BasicLogger
import numpy as np
import flgo.simulator.base as ss

class HierLogger(BasicLogger):
    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        test_metric = self.coordinator.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        # output to stdout
        self.show_current_output()