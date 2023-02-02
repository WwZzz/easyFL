import utils.fflow as flw
import config as cfg
import multiprocessing as mp

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
    mp.set_start_method('spawn')
    # mp.set_sharing_strategy('file_system')
    main()