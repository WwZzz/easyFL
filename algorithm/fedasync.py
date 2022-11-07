"""This is a non-official implementation of 'Asynchronous Federated Optimization' (http://arxiv.org/abs/1903.03934). """
from .fedbase import BasicServer
from .fedprox import Client
import utils.system_simulator as ss
import utils.fflow as flw

class Server(BasicServer):
    def __init__(self, option, model, clients, test_data = None):
        super(Server, self).__init__(option, model, clients, test_data)
        self.init_algo_para({'period':1, 'alpha': 0.6, 'mu':0.005, 'flag':'constant', 'hinge_a':10, 'hinge_b':6, 'poly_a':0.5})
        self.asynchronous = True
        self.tolerance_for_latency = 1000
        self.client_taus = [0 for _ in self.clients]
        self.updated = True

    def run(self):
        flw.logger.time_start('Total Time Cost')
        while self.current_round <= self.num_rounds:
            # using logger to evaluate the model
            if self.updated:
                self.updated = False
                flw.logger.info("--------------Round {}--------------".format(self.current_round))
                # check log interval
                if flw.logger.check_if_log(self.current_round, self.eval_interval):
                    flw.logger.time_start('Eval Time Cost')
                    flw.logger.log_once()
                    flw.logger.time_end('Eval Time Cost')
                # check if early stopping
                if flw.logger.early_stop(): break
            self.iterate()
            # decay learning rate
            self.global_lr_scheduler(self.current_round)
        flw.logger.info("--------------Final Evaluation--------------")
        flw.logger.time_start('Eval Time Cost')
        flw.logger.log_once()
        flw.logger.time_end('Eval Time Cost')
        flw.logger.info("=================End==================")
        flw.logger.time_end('Total Time Cost')
        # save results as .json file
        flw.logger.save_output_as_json()
        return

    @ss.time_step
    def iterate(self):
        # Scheduler periodically triggers the idle clients to locally train the model
        self.selected_clients = self.sample() if (ss.clock.current_time%self.period)==0 else []
        if len(self.selected_clients)>0:
            flw.logger.info('Select clients {} at time {}'.format(self.selected_clients, ss.clock.current_time))
        # Record the timestamp of the selected clients
        for cid in self.selected_clients: self.client_taus[cid] = self.current_round
        # Check the currently received models
        res = self.communicate(self.selected_clients)
        received_models = res['model']
        received_client_ids = res['__cid']
        if len(received_models) > 0:
            flw.logger.info('Receive new models from clients {} at time {}'.format(received_client_ids, ss.clock.current_time))
            # averaging the simultaneously received models at the current moment
            taus = [self.client_taus[cid] for cid in received_client_ids]
            alpha_ts = [self.alpha * self.s(self.current_round - tau) for tau in taus]
            currently_updated_models = [(1-alpha_t)*self.model+alpha_t*model_k for alpha_t, model_k in zip(alpha_ts, received_models) ]
            self.model = self.aggregate(currently_updated_models)
            # update aggregation round and the flag `updated`
            self.current_round += 1
            self.updated = True
        return

    def s(self, tau):
        t = ss.clock.current_time
        if self.flag == 'constant':
            return 1
        elif self.flag == 'hinge':
            return 1 if t-tau <= self.b else 1.0/(self.a * (t-tau-self.b))
        elif self.flag == 'poly':
            return (t-tau+1)**(-self.a)
