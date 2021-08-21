import utils.fflow as flw
import ujson

def main():
    # read options
    option = flw.read_option()
    # set random seed
    flw.setup_seed(option['seed'])
    # initialize server
    server = flw.initialize(option)
    # start federated optimization
    output = server.run()
    # save results as .json file
    with open('fedtask/'+ option['task'] + '/record/' + flw.output_filename(option, server), 'w') as outfile:
        ujson.dump(output, outfile)

if __name__ == '__main__':
    main()



