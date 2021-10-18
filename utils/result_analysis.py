"""
the rules of naming records
M: model
R: communication round
B: batch size
E: local epoch
LR: learning rate (step size)
P: the proportion of selected clients in each round
S: random seed
T: the proportion of training part of clients' data
LD: learning rate scheduler + learning rate decay
WD: weight decay
DR: the degree of dropout of clients
"""
# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import ujson
import prettytable as pt
import os
import numpy as np

def read_data_into_dicts(task, records):
    path = '../fedtask/'+task+'/record'
    files = os.listdir(path)
    res = []
    for f in records:
        if f in files:
            file_path = os.path.join(path, f)
            with open(file_path, 'r') as inf:
                rec = ujson.load(inf)
            res.append(rec)
    return res

def draw_curve(dicts, curve='train_losses', legends = []):
    # plt.figure(figsize=(100,100), dpi=100)
    if not legends: legends = [d['meta']['method'] for d in dicts]
    for i,dict in enumerate(dicts):
        num_rounds = dict['meta']['num_rounds']
        eval_interval = dict['meta']['eval_interval']
        x = []
        for round in range(num_rounds + 1):
            if eval_interval > 0 and (round == 0 or round % eval_interval == 0 or round == num_rounds):
                x.append(round)
        if curve == 'train_losses':
            y = [dict[curve][round] for round in range(num_rounds + 1) if (round == 0 or round % eval_interval == 0 or round == num_rounds)]
        else:
            y = dict[curve]
        plt.plot(x, y, label=legends[i], linewidth=1)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
    return

def filename_filter(fnames=[], filter={}):
    if filter:
        for key in filter.keys():
            con = filter[key].strip()
            if con[0] in ['[','{','(']:
                con = 'in ' + con
            elif '0'<=con[0]<='9' or con[0]=='.' or con[0]=='-':
                con = '==' + con
            elif 'a'<=con[0]<='z' or 'A'<=con[0]<='Z':
                con = "'"+con+"'"
            fnames = [f for f in fnames if eval(f[f.find('_'+key)+len(key)+1:f.find('_',f.find(key)+1)]+' '+con)]
    return fnames

def round_to_achieve_test_acc(records, dicts, target=0):
    tb= pt.PrettyTable()
    tb.field_names = [
        'Record',
        'Round to Achieve {}% Test-Acc.'.format(target),
    ]
    for rec, d in zip(records, dicts):
        r = -1
        for i in range(len(d['test_accs'])):
            if d['test_accs'][i]>=target-0.000001:
                r = i*d['meta']['eval_interval']
                break
        tb.add_row([rec, r])
    print(tb)
    return

def scan_records(task, header = '', filter = {}):
    path = '../fedtask/' + task + '/record'
    files = os.listdir(path)
    # check headers
    files = [f for f in files if f.startswith(header+'_')]
    return filename_filter(files, filter)

def print_table(records, dicts):
    tb = pt.PrettyTable()
    tb.field_names = [
        'Record',
        'Test-Acc.',
        'Valid-Acc.',
        'Train-Loss',
        'Test-Loss',
        'Best Test-Acc./Round',
        'Highest Valid-Acc.',
        'Lowest Valid-Acc.',
        'Mean-Valid-Acc.',
        'Var-Valid-Acc.',
    ]
    for rec,d in zip(records, dicts):
        testacc  = d['test_accs'][-1]
        validacc = d['mean_curve'][-1]
        trainloss = d['train_losses'][-1]
        testloss = d['test_losses'][-1]
        bestacc = 0
        idx = -1
        for i in range(len(d['test_accs'])):
            if d['test_accs'][i]>bestacc:
                bestacc = d['test_accs'][i]
                idx = i*d['meta']['eval_interval']
        highest = float(np.max(d['acc_dist']))
        lowest = float(np.min(d['acc_dist']))
        mean_valid = float(np.mean(d['acc_dist']))
        var_valid = float(np.std(d['acc_dist']))
        tb.add_row([rec, testacc, validacc, trainloss, testloss, str(bestacc)+'/'+str(idx), highest, lowest, mean_valid, var_valid])
    tb.sortby = 'Test-Acc.'
    tb.reversesort = True
    print(tb)

if __name__ == '__main__':
    # task+record
    task = 'mnist_client100_dist0_beta0_noise0'
    headers = [
        'fedavg',
        # 'fedprox',
    ]
    flt = {
        # 'E': '5',
        # 'B': '10',
        # 'LR': '>0.15',
        # 'R': '30',
        # 'P': '0.0',
        # 'S': '0',
    }
    # read and filter the filenames
    records = set()
    for h in headers:
        records = records.union(set(scan_records(task, h, flt)))
    records = list(records)
    # read the selected files into dicts
    dicts = read_data_into_dicts(task, records)

    # print table
    print_table(records, dicts)

    # draw curves
    curve_names = [
        'train_losses',
        'test_losses',
        'test_accs',
    ]
    # create legends
    legends = [records[i] for i in range(len(records))]
    for curve in curve_names:
        draw_curve(dicts, curve, legends)
        plt.title(task)
        plt.xlabel("communication rounds")
        plt.ylabel(curve)
        ax = plt.gca()
        plt.grid()
        plt.show()