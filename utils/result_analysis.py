"""
the rules of naming records
M: model
R: communication round
B: batch size
E: local epoch
NS: number of local update steps
LR: learning rate (step size)
P: the proportion of selected clients in each round
S: random seed
LD: learning rate scheduler + learning rate decay
WD: weight decay
DR: the degree of dropout of clients
AC: the active rate of clients
"""
# import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

import matplotlib.pyplot as plt
import json
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
                s_inf = inf.read()
                rec = json.loads(s_inf)
            res.append(rec)
    return res

def draw_curve(dicts, curve='train_loss', legends = [], final_round = -1):
    # plt.figure(figsize=(100,100), dpi=100)
    if not legends: legends = [d['meta']['algorithm'] for d in dicts]
    for i,dict in enumerate(dicts):
        num_rounds = dict['meta']['num_rounds']
        eval_interval = dict['meta']['eval_interval']
        x = []
        for round in range(num_rounds + 1):
            if eval_interval > 0 and (round == 0 or round % eval_interval == 0 or round == num_rounds):
                x.append(round)
        y = dict[curve]
        plt.plot(x, y, label=legends[i], linewidth=1)
        if final_round>0: plt.xlim((0, final_round))
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
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
            res = []
            for f in fnames:
                if f.find('_' + key)==-1: continue
                if eval(f[f.find('_' + key) + len(key) + 1:f.find('_', f.find('_' + key) + 1)] + ' ' + con):
                    res.append(f)
            fnames = res
    return fnames

def scan_records(task, header = '', filter = {}):
    path = '../fedtask/' + task + '/record'
    files = os.listdir(path)
    # check headers
    files = [f for f in files if f.startswith(header+'_') and f.endswith('.json')]
    return filename_filter(files, filter)

def get_key_from_filename(record, key = ''):
    if key=='': return ''
    value_start = record.find('_'+key)+len(key)+1
    value_end = record.find('_',value_start)
    return record[value_start:value_end]

def create_legend(records=[], keys=[]):
    if records == []:
        return records
    if len(keys)==0:
        return [rec[:rec.find('_M')] for rec in records]
    res = []
    for rec in records:
        s = [rec[:rec.find('_M')]]
        values = [k+get_key_from_filename(rec, k) for k in keys]
        s.extend(values)
        res.append(" ".join(s))
    return res

# task-specific analysis tools, which should be overwrited if used to show additional information
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
        testacc  = d['test_accuracy'][-1]
        validacc = d['mean_valid_accuracy'][-1]
        trainloss = d['train_loss'][-1]
        testloss = d['test_loss'][-1]
        best_testacc = 0
        idx = -1
        for i in range(len(d['test_accuracy'])):
            if d['test_accuracy'][i]>best_testacc:
                best_testacc = d['test_accuracy'][i]
                idx = i*d['meta']['eval_interval']
        highest = float(np.max(d['valid_accuracy'][-1]))
        lowest = float(np.min(d['valid_accuracy'][-1]))
        mean_valid = float(np.mean(d['valid_accuracy'][-1]))
        var_valid = float(np.std(d['valid_accuracy'][-1]))
        tb.add_row([rec, testacc, validacc, trainloss, testloss, str(best_testacc)+'/'+str(idx), highest, lowest, mean_valid, var_valid])
    tb.sortby = 'Test-Acc.'
    tb.reversesort = True
    print(tb)

def round_to_achieve_test_acc(records, dicts, target=0):
    tb= pt.PrettyTable()
    tb.field_names = [
        'Record',
        'Round to Achieve {}% Test-Acc.'.format(target),
    ]
    for rec, d in zip(records, dicts):
        r = -1
        for i in range(len(d['test_accuracy'])):
            if d['test_accuracy'][i]>=target-0.000001:
                r = i*d['meta']['eval_interval']
                break
        tb.add_row([rec, r])
    print(tb)
    return

if __name__ == '__main__':
    # task+record
    task = 'mnist_classification_cnum100_dist0_skew0_seed0'
    headers = [
        'fedavg',
    ]
    curve_names = [
        'train_loss',
        'test_loss',
        'test_accuracy',
    ]
    # exclude records whose hyper-parameters not satisfied the constraint defined in flt below
    # For example, 1) if flt['B']=='64', then only records with batch_size==64 will be preserved
    #              2)if flt['LR']=='<0.1', then only records with learning_rate<=0.1 will be preserved
    flt = {
        # 'B': '64',
        # 'NS':'50',
        # 'E': '1',
        # 'LR': '0.01',
        # 'R': '30',
        # 'P': '0.01',
        # 'S': '0',
    }
    # read and filter the filenames
    records = set()
    for h in headers:records = records.union(set(scan_records(task, h, flt)))
    records = list(records)
    # read the selected files into dicts
    dicts = read_data_into_dicts(task, records)

    # print table
    print_table(records, dicts)
    # create legends with selected parameters specified by the second parameter of create_legend()
    legends = create_legend(records, [])
    # draw curves
    for curve in curve_names:
        draw_curve(dicts, curve, legends)
        plt.title(task)
        # plt.xlabel("communication rounds")
        plt.ylabel(curve)
        ax = plt.gca()
        plt.grid()
        try:
            plt.show()
        except:
            print('Failed to call plt.show()')
        finally:
            import datetime
            res_name = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H:%M:%S')
            res_name = res_name + curve +'.png'
            plt.savefig(os.path.join('../fedtask', task, 'record', res_name))