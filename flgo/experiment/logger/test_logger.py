from flgo.experiment.logger import BasicLogger

class TestLogger(BasicLogger):
    """This logger only records metrics on global testing dataset"""
    def log_once(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        self.show_current_output()