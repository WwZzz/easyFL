import utils.fflow as flw

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