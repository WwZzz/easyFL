from flgo.experiment.logger import BasicLogger
import numpy as np

class GValLogger(BasicLogger):
    def log_once(self, *args, **kwargs):
        val_metric = self.server.test(flag='val')
        for met_name, met_val in val_metric.items():
            self.output['val_' + met_name].append(met_val)
        self.show_current_output()