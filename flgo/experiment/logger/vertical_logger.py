from flgo.experiment.logger import BasicLogger

class VerticalLogger(BasicLogger):
    r"""Vertical Logger. Vertical FL should use this Logger"""
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local_movielens_recommendation data size)"""
        for c in self.participants:
            self.output['client_datavol'].append(len(c.train_data))

    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        self.info('Current_time:{}'.format(self.clock.current_time))
        self.output['time'].append(self.clock.current_time)
        test_metric = self.coordinator.test('test')
        val_metric = self.coordinator.test('val')
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        for met_name, met_test in val_metric.items():
            self.output['val_' + met_name].append(met_val)
        self.show_current_output()