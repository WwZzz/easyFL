import copy
from tqdm import trange
from flgo.algorithm.fedbase import BasicServer
from flgo.algorithm.fedbase import BasicClient

class Server(BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'ft_lr':0.1, 'ft_epochs':5, 'ft_batchsize':50, 'ft_weight_decay':0.0, 'ft_momentum':0.0, 'tune_key':'loss', 'larger_is_better':False})

    def run(self):
        """
        Running the FL symtem where the global model is trained and evaluated iteratively.
        """
        self.gv.logger.time_start('Total Time Cost')
        if self.eval_interval>0:
            # evaluating initial model performance
            self.gv.logger.info("--------------Initial Evaluation--------------")
            self.gv.logger.time_start('Eval Time Cost')
            self.gv.logger.log_once()
            self.gv.logger.time_end('Eval Time Cost')
        while self.current_round <= self.num_rounds:
            self.gv.clock.step()
            # iterate
            updated = self.iterate()
            # using logger to evaluate the model if the model is updated
            if updated is True or updated is None:
                self.gv.logger.info("--------------Round {}--------------".format(self.current_round))
                # check log interval
                if self.gv.logger.check_if_log(self.current_round, self.eval_interval):
                    self.gv.logger.time_start('Eval Time Cost')
                    self.gv.logger.log_once()
                    self.gv.logger.time_end('Eval Time Cost')
                # check if early stopping
                if self.gv.logger.early_stop(): break
                self.current_round += 1
                # decay learning rate
                self.global_lr_scheduler(self.current_round)
        self.gv.logger.info("=================Federated Training End==================")
        self.gv.logger.time_end('Total Time Cost')
        self.gv.logger.info("=================Local Model Fine Tuning==================")
        for c in self.clients:
            c.finetune()
        self.gv.logger.info("--------------After Tuning--------------".format(self.current_round))
        # check log interval
        self.gv.logger.time_start('Eval Time Cost')
        self.gv.logger.log_once()
        self.gv.logger.time_end('Eval Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()

class Client(BasicClient):
    def finetune(self):
        if self.model is None:
            self.model = copy.deepcopy(self.server.model)
        dataloader = self.calculator.get_dataloader(self.train_data, batch_size=self.ft_batchsize)
        optimizer = self.calculator.get_optimizer(self.model, lr=self.ft_lr, weight_decay=self.ft_weight_decay, momentum=self.ft_momentum)
        epoch_iter = trange(self.ft_epochs+1)
        op_model_dict = copy.deepcopy(self.model.state_dict())
        op_met = self.test(self.model, 'val')
        op_epoch = 0
        for e in epoch_iter:
            if e<self.ft_epochs:
                for batch_id, batch_data in enumerate(dataloader):
                    optimizer.zero_grad()
                    batch_data = self.calculator.to_device(batch_data)
                    loss = self.calculator.compute_loss(self.model, batch_data)['loss']
                    loss.backward()
                    optimizer.step()
                val_metric = self.test(self.model, 'val')
                if (self.larger_is_better and val_metric[self.tune_key]>op_met[self.tune_key]) or ((not self.larger_is_better) and val_metric[self.tune_key]<op_met[self.tune_key]):
                    op_met = val_metric
                    op_epoch = e+1
                    op_model_dict = copy.deepcopy(self.model.state_dict())
                terms = ["Client {}".format(self.id), "Epoch {}/{}".format(e+1, self.ft_epochs)]
                terms.extend(['val_{}: {:.4f}'.format(k,v) for k,v in val_metric.items()])
                epoch_iter.set_description("\t".join(terms))
            else:
                terms = ["Client {}".format(self.id), "Optimal Epoch {}/{}".format(op_epoch, self.ft_epochs)]
                terms.extend(['val_{}: {:.4f}'.format(k, v) for k, v in op_met.items()])
                epoch_iter.set_description("\t".join(terms))
                self.model.load_state_dict(op_model_dict)
        return