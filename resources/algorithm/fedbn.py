"""
This is a non-official implementation of 'FedBN: Federated Learning on Non-IID Features via Local Batch Normalization'
(https://openreview.net/pdf?id=6YEQUn0QICG). The official implementation is at 'https://github.com/med-air/FedBN'
"""

import flgo.algorithm.fedbase as fa

Server = fa.BasicServer

class Client(fa.BasicClient):
    def unpack(self, received_pkg):
        """Preserve the BN layers when receiving the global model from the server. The BN module should be claimed with the keyword 'bn'."""
        global_model = received_pkg['model']
        if self.model==None:
            self.model = global_model
        else:
            new_dict = self.model.state_dict()
            global_dict = global_model.state_dict()
            for key in self.model.state_dict().keys():
                if 'bn' in key.lower() or 'batch_norm' in key.lower() or 'batchnorm' in key.lower(): continue
                new_dict[key] = global_dict[key]
            self.model.load_state_dict(new_dict)
        return self.model