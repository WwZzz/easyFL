from flgo.experiment.logger import BasicLogger
import numpy as np

class FullLogger(BasicLogger):
    def log_once(self, *args, **kwargs):
        test_metric = self.server.test()
        for met_name, met_val in test_metric.items():
            self.output['test_' + met_name].append(met_val)
        val_metric = self.server.test(flag='val')
        for met_name, met_val in val_metric.items():
            self.output['val_' + met_name].append(met_val)
        # calculate weighted averaging of metrics on training datasets across participants
        local_data_vols = [c.datavol for c in self.clients]
        total_data_vol = sum(local_data_vols)
        train_metrics = self.server.global_test(flag='train')
        for met_name, met_val in train_metrics.items():
            self.output['train_' + met_name + '_dist'].append(met_val)
            self.output['train_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
        # calculate weighted averaging and other statistics of metrics on validation datasets across clients
        local_val_metrics = self.server.global_test(flag='val')
        for met_name, met_val in local_val_metrics.items():
            self.output['local_val_'+met_name+'_dist'].append(met_val)
            self.output['local_val_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_val_' + met_name].append(np.mean(met_val))
            self.output['std_local_val_' + met_name].append(np.std(met_val))
        local_test_metrics = self.server.global_test(flag='test')
        for met_name, met_val in local_test_metrics.items():
            self.output['local_test_'+met_name+'_dist'].append(met_val)
            self.output['local_test_' + met_name].append(1.0 * sum([client_vol * client_met for client_vol, client_met in zip(local_data_vols, met_val)]) / total_data_vol)
            self.output['mean_local_test_' + met_name].append(np.mean(met_val))
            self.output['std_local_test_' + met_name].append(np.std(met_val))
        # output to stdout
        self.show_current_output()