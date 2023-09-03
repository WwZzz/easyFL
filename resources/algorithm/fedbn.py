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
            for key in self.model.state_dict().keys():
                if 'bn' not in key.lower():
                    self.model.state_dict()[key].data.copy_(global_model.state_dict()[key])
        return self.model