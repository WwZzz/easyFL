import utils.fflow as flw
import torch
import config as cfg

def main():
    # read options
    option = flw.read_option()
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server, clients and fedtask
    coordinator = flw.initialize(option)
    # start federated optimization
    try:
        coordinator.run()
    except:
        # log the exception that happens during training-time
        cfg.logger.exception("Exception Logged")
        raise RuntimeError

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()