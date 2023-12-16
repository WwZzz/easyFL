from flgo.experiment.logger import BasicLogger
import numpy as np

class PFLLogger(BasicLogger):
    def log_once(self, *args, **kwargs):
        # local performance
        cvals = []
        ctests = []
        for c in self.clients:
            model = c.model if (hasattr(c, 'model') and c.model is not None) else self.server.model
            cvals.append(c.test(model, 'val'))
            ctests.append(c.test(model, 'test'))
        cval_dict = {}
        if len(cvals)>0:
            for met_name in cvals[0].keys():
                if met_name not in cval_dict.keys(): cval_dict[met_name] = []
                for cid in range(len(cvals)):
                    cval_dict[met_name].append(cvals[cid][met_name])
                self.output['val_'+met_name].append(float(np.array(cval_dict[met_name]).mean()))
        ctest_dict = {}
        if len(ctests)>0:
            for met_name in ctests[0].keys():
                if met_name not in ctest_dict.keys(): ctest_dict[met_name] = []
                for cid in range(len(ctests)):
                    ctest_dict[met_name].append(ctests[cid][met_name])
                self.output['local_test_'+met_name].append(float(np.array(ctest_dict[met_name]).mean()))
        # global performance
        # gmetrics = self.server.test(self.server.model, 'test')
        # for met_name, met_val in gmetrics.items():
        #     self.output['global_test_' + met_name].append(met_val)
        # output to stdout
        self.show_current_output()