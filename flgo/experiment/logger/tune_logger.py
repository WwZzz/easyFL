from flgo.experiment.logger import BasicLogger

import numpy as np

class TuneLogger(BasicLogger):
    """This logger only records metrics on validation dataset"""
    def log_once(self, *args, **kwargs):
        if self.scene == 'horizontal':
            val_metrics = self.coordinator.global_test(flag='val')
            local_data_vols = [c.datavol for c in self.participants]
            total_data_vol = sum(local_data_vols)
            for met_name, met_val in val_metrics.items():
                self.output['val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        elif self.scene == 'vertical':
            val_metrics = self.coordinator.test('val')
            for met_name, met_val in val_metrics.items():
                self.output['val_' + met_name].append(met_val)
        # output to stdout
        self.show_current_output()