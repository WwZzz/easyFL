from utils.tools import *
import json

def main():
    # read options
    option = read_option()
    # set random seed
    setup_seed(option['seed'])
    # initialize
    server = initialize(option)
    outdict = server.run()
    # save results as method{}_mpara{}_r{}_b{}_e{}_lr{}_p{}_seed{}.json file
    filename=output_filename(option, server)
    with open('task/'+ option['dataset'] + '/record/' + filename, 'w') as outfile:
        json.dump(outdict, outfile)

if __name__ == '__main__':
    main()



