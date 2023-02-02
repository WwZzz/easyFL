import experiment.logger.basic_logger as bl
import numpy as np
import system_simulator.base as ss

class Logger(bl.Logger):
    def initialize(self):
        """This method is used to record the stastic variables that won't change across rounds (e.g. local data size)"""
        for c in self.participants:
            self.output['client_datavol'].append(len(c.train_data))

    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        self.info('Current_time:{}'.format(self.clock.current_time))
        valid_metrics = self.coordinator.global_test('valid')
        local_data_vols = [c.datavol for c in self.participants]
        total_data_vol = sum(local_data_vols)
        for met_name, met_val in valid_metrics.items():
            self.output['valid_'+met_name+'_dist'].append(met_val)
            self.output['valid_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_valid_' + met_name].append(np.mean(met_val))
            self.output['std_valid_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()