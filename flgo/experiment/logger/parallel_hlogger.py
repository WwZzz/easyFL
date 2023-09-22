from flgo.experiment.logger import BasicLogger

class ParallelHFLLogger(BasicLogger):
    def log_once(self, *args, **kwargs):
        test_metrics = self.object.test()
        for met, val in test_metrics.items():
            self.output[self.object.name+'_test_'+met].append(val)
        self.show_current_output()