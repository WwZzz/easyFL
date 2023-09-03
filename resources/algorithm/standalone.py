import copy
from tqdm import trange
import flgo.algorithm.fedbase as fab

class Server(fab.BasicServer):
    def initialize(self, *args, **kwargs):
        self.init_algo_para({'tune_key':'loss', 'larger_is_better': False})

    def run(self):
        self.gv.logger.time_start('Total Time Cost')
        # evaluating initial model performance
        self.gv.logger.info("--------------Initial Evaluation--------------")
        self.gv.logger.time_start('Eval Time Cost')
        self.gv.logger.log_once()
        self.gv.logger.time_end('Eval Time Cost')
        # Standalone Training
        for c in self.clients: c.finetune()
        self.gv.logger.info("=================End==================")
        self.gv.logger.time_end('Total Time Cost')
        self.gv.logger.time_start('Eval Time Cost')
        self.gv.logger.log_once()
        self.gv.logger.time_end('Eval Time Cost')
        # save results as .json file
        self.gv.logger.save_output_as_json()
        return

class Client(fab.BasicClient):
    def initialize(self, *args, **kwargs):
        self.model = copy.deepcopy(self.server.model)

    def finetune(self):
        dataloader = self.calculator.get_dataloader(self.train_data, batch_size=self.batch_size)
        optimizer = self.calculator.get_optimizer(self.model, lr=self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
        epoch_iter = trange(self.num_epochs+1)
        op_model_dict = copy.deepcopy(self.model.state_dict())
        op_met = self.test(self.model, 'val')
        op_epoch = 0
        for e in epoch_iter:
            if e<self.num_epochs:
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
                terms = ["Client {}".format(self.id), "Epoch {}/{}".format(e+1, self.num_epochs)]
                terms.extend(['val_{}: {:.4f}'.format(k,v) for k,v in val_metric.items()])
                epoch_iter.set_description("\t".join(terms))
            else:
                terms = ["Client {}".format(self.id), "Optimal Epoch {}/{}".format(op_epoch, self.num_epochs)]
                terms.extend(['val_{}: {:.4f}'.format(k, v) for k, v in op_met.items()])
                epoch_iter.set_description("\t".join(terms))
                self.model.load_state_dict(op_model_dict)
        return


