import utils.fflow as flw
import numpy as np

class MyLogger(flw.Logger):
    # def log(self, server=None):
    #     if server==None: return
    #     if server.option['task'].startswith('quad2d'):
    #         if self.output == {}:
    #             self.output = {
    #                 "meta": server.option,
    #                 "mean_curve": [],
    #                 "var_curve": [],
    #                 "train_losses": [],
    #                 "test_accs": [],
    #                 "test_losses": [],
    #                 "valid_accs": [],
    #                 "client_accs": {},
    #                 "mean_valid_accs": [],
    #                 "position": [],
    #                 "local_optimal": [],
    #                 "drop_rates": [],
    #             }
    #             self.output['global_optimal'] = server.test_data[0].tolist()
    #             for c in server.clients:
    #                 self.output['local_optimal'].append(c.valid_data[0].tolist())
    #                 self.output['drop_rates'].append(c.drop_rate)
    #         position = []
    #         for i, p in enumerate(server.model.parameters()):
    #             position.append(p.tolist()[0])
    #         test_metric, test_loss = server.test()
    #         valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
    #         train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')
    #         self.output['train_losses'].append(
    #             1.0 * sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)]) / server.data_vol)
    #         self.output['valid_accs'].append(valid_metrics)
    #         self.output['test_accs'].append(test_metric)
    #         self.output['test_losses'].append(test_loss)
    #         self.output['mean_valid_accs'].append(
    #             1.0 * sum([ck * acc for ck, acc in zip(server.client_vols, valid_metrics)]) / server.data_vol)
    #         self.output['mean_curve'].append(np.mean(valid_metrics))
    #         self.output['var_curve'].append(np.std(valid_metrics))
    #         self.output['position'].append(position[-1])
    #         for cid in range(server.num_clients):
    #             self.output['client_accs'][server.clients[cid].name] = [self.output['valid_accs'][i][cid] for i in
    #                                                                     range(len(self.output['valid_accs']))]
    #         print(self.temp.format("Training Loss:", self.output['train_losses'][-1]))
    #         print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
    #         print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
    #         print(self.temp.format("Validating Accuracy:", self.output['mean_valid_accs'][-1]))
    #         print(self.temp.format("Mean of Client Accuracy:", self.output['mean_curve'][-1]))
    #         print(self.temp.format("Std of Client Accuracy:", self.output['var_curve'][-1]))
    #         print(self.temp.format("Virtual Time Cost:", server.virtual_clock[-1]))
    #         print("Current position: {:.4f}, {:.4f}".format(position[-1][0], position[-1][1]))
    #     else:
    #         if self.output == {}:
    #             self.output = {
    #                 "meta":server.option,
    #                 "test_accs":[],
    #                 "test_losses":[],
    #             }
    #         test_metric, test_loss = server.test()
    #         self.output['test_accs'].append(test_metric)
    #         self.output['test_losses'].append(test_loss)
    #         print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
    #         print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
    def log(self, server=None):
        if self.output == {}:
            self.output = {
                "meta": server.option,
                "mean_curve": [],
                "var_curve": [],
                "train_losses": [],
                "test_accs": [],
                "test_losses": [],
                "valid_accs": [],
                "client_accs": {},
                "mean_valid_accs": [],
                "drop_rates": [],
                "ac_rates":[]
            }
            for c in server.clients:
                self.output['drop_rates'].append(c.drop_rate)
                self.output['ac_rates'].append(c.active_rate)
        test_metric, test_loss = server.test()
        valid_metrics, valid_losses = server.test_on_clients(self.current_round, 'valid')
        train_metrics, train_losses = server.test_on_clients(self.current_round, 'train')
        self.output['train_losses'].append(
            1.0 * sum([ck * closs for ck, closs in zip(server.client_vols, train_losses)]) / server.data_vol)
        self.output['valid_accs'].append(valid_metrics)
        self.output['test_accs'].append(test_metric)
        self.output['test_losses'].append(test_loss)
        self.output['mean_valid_accs'].append(
            1.0 * sum([ck * acc for ck, acc in zip(server.client_vols, valid_metrics)]) / server.data_vol)
        self.output['mean_curve'].append(np.mean(valid_metrics))
        self.output['var_curve'].append(np.std(valid_metrics))
        for cid in range(server.num_clients):
            self.output['client_accs'][server.clients[cid].name] = [self.output['valid_accs'][i][cid] for i in
                                                                    range(len(self.output['valid_accs']))]
        print(self.temp.format("Training Loss:", self.output['train_losses'][-1]))
        print(self.temp.format("Testing Loss:", self.output['test_losses'][-1]))
        print(self.temp.format("Testing Accuracy:", self.output['test_accs'][-1]))
        print(self.temp.format("Validating Accuracy:", self.output['mean_valid_accs'][-1]))
        print(self.temp.format("Mean of Client Accuracy:", self.output['mean_curve'][-1]))
        print(self.temp.format("Std of Client Accuracy:", self.output['var_curve'][-1]))

logger = MyLogger()

def main():
    # read options
    option = flw.read_option()
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    server.run()

if __name__ == '__main__':
    main()




