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
NET: the config of network condition
CMP: the config of computing resource condition
"""
import argparse
import datetime
import random
import matplotlib as mpl
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
import json
import prettytable as pt
import os
import numpy as np
import collections
import yaml
import math

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

def get_communication_round_from_rec(record):
    num_rounds = record['meta']['num_rounds']
    eval_interval = record['meta']['eval_interval']
    x = []
    for round in range(num_rounds + 1):
        if eval_interval > 0 and (round == 0 or round % eval_interval == 0 or round == num_rounds):
            x.append(round)
        if record['meta']['early_stop']>0 and len(x) >= len(record['valid_loss']):
            break
    return x

def draw_curves_from_records(records, curve='train_loss'):
    max_x = -1
    for rec in records:
        dict = records[rec]
        x = get_communication_round_from_rec(dict)
        y = dict[curve]
        plt.plot(x, y, label=dict['legend'], linewidth=1)
        max_x = x[-1] if x[-1]>max_x else max_x
    plt.legend()
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=1)
    return

def draw_curve_with_range(x, y, y_bottom, y_top, legend='', color='', ax=None):
    assert len(y) == len(y_bottom) and len(y_bottom) == len(y_top)
    if ax==None: ax=plt
    ax.plot(x, y, color=color, label=legend)
    ax.fill_between(x, y_top, y_bottom, color=color, alpha=0.3)
    ax.legend()

def filename_filter(fnames=[], filter={}):
    if filter:
        for key in filter.keys():
            con = filter[key].strip()
            if con[0] in ['[','{','(']:
                con = 'in ' + con
            elif '0'<=con[0]<='9' or con[0]=='.' or con[0]=='-':
                con = '==' + con
            elif 'a'<=con[0]<='z' or 'A'<=con[0]<='Z':
                con = "=='"+con+"'"
            res = []
            for f in fnames:
                if f.find('_' + key)==-1: continue
                fv = f[f.find('_' + key) + len(key) + 1:f.find('_', f.find('_' + key) + 1)]
                if 'a'<=fv[0]<='z' or 'A'<=fv[0]<='Z': fv = "'"+fv+"'"
                if eval(fv + ' ' + con):
                    res.append(f)
            fnames = res
    return fnames

def scan_records(task, header = '', filter = {}):
    path = '../fedtask/' + task + '/record'
    files = os.listdir(path)
    # check headers
    files = [f for f in files if f.startswith(header) and f.endswith('.json')]
    return filename_filter(files, filter)

def get_key_from_record_name(record_name, key =''):
    if key=='': return ''
    value_start = record_name.find('_' + key) + len(key) + 1
    value_end = record_name.find('_', value_start)
    return record_name[value_start:value_end]

def set_legend(records={}, legend_filter=[]):
    if records == {}:return records
    for rec in records:
        records[rec]['legend'] = []
        records[rec]['legend'].append(rec[:rec.find('_M')])
    for key in legend_filter:
        for rec in records:
            val = key + get_key_from_record_name(rec, key)
            records[rec]['legend'].append(val)
    for rec in records:
        records[rec]['legend'] = " ".join(records[rec]['legend'])
    return

def group_records_ignoring_seed(records={}):
    grouped_recs = collections.defaultdict(list)
    for rec in records:
        name = rec[:rec.find('S')] + rec[rec.find('LD'):]
        grouped_recs[name].append(records[rec])
    return grouped_recs

def func_on_dicts(dicts, key='test_loss', func=None, option={}):
    if func is None: return None
    res = []
    for d in dicts:
        res.append(d[key])
    res = func(res, **option)
    return res

def statistic_on_dicts(dicts, key='test_loss', name='mean', option={'axis':0}):
    func_dict={'mean':np.mean, 'std':np.std, 'min':np.min, 'max':np.max}
    def f(x, axis=0):
        return func_dict[name](np.array(x), axis=axis).tolist()
    return func_on_dicts(dicts, key, f, option)

def cfg_to_records(config):
    cfg_template = {
        'flt': {},
        'legend_flt': [],
        'ploter': {},
        'info': {},
    }
    # check cfg
    cfg_keys = config.keys()
    if 'task' not in cfg_keys or 'header' not in cfg_keys:
        raise NotImplementedError
    for k in cfg_template:
        if k not in cfg_keys:
            config[k] = cfg_template[k]
    task, header, flt = config['task'], config['header'], config['flt']
    rec_names = []
    for h in header: rec_names.extend(scan_records(task, h, flt))
    rec_names = list(rec_names)
    dicts = read_data_into_dicts(task, rec_names)
    records = {}
    for rec, dict in zip(rec_names, dicts):
        records[rec] = dict
    return records

# Analyser
class Analyser:
    def __init__(self, records):
        self.records = records
        self.grouped_records = group_records_ignoring_seed(records)
        self.rec_names = [r for r in records]
        self.rec_dicts = [v for v in records.values()]
        self.task = self.rec_dicts[0]['meta']['task']
        self.save_path = os.path.join('..','fedtask',self.task,'record')
        for rec in records:
            records[rec]['communication_round'] = get_communication_round_from_rec(records[rec])
            records[rec]['client_id'] = [cid for cid in range(int(get_key_from_record_name(records[rec]['meta']['task'], 'cnum')))]

class Drawer(Analyser):
    _axes_keywords = ['xlim','ylim','axis','xlabel','ylabel','title',]
    _default_option = {
        'plot': ['linestyle', 'marker', 'linewidth', 'markersize',],
        'scatter': ['s', 'cmap', 'alpha', 'color',],
        'group_plot': ['linestyle', 'marker', 'linewidth', 'markersize',],

    }
    def __init__(self, records, save_figure=False):
        super().__init__(records)
        self.colors = [c for c in mpl.colors.CSS4_COLORS.keys()]
        random.shuffle(self.colors)
        self.save_figure = save_figure
        self.default_option = {
            'plot': {
                'linestyle': '-',
                'marker':'o',
                'linewidth':1,
                'markersize':1,
            },
            'scatter':{
                's':1,
                'cmap': 'viridis',
                'alpha':0,
                'color':'r',
            },
        }

    def load_ploter_option(self, funcname, plot_obj):
        default_keys = [k for k in self.default_option[funcname].keys()]
        res_option = {}
        for key in plot_obj.keys():
            if key in default_keys:
                res_option[key] = plot_obj[key]
        return res_option

    def load_axes_option(self, plot_obj):
        axes_keys = [k for k in plot_obj.keys() if k in self._axes_keywords]
        for key in axes_keys:
            f = eval('plt.'+key)
            f(plot_obj[key])

    def get_current_axes(self, plot_obj, id):
        if '_axes' in plot_obj.keys():
            return plot_obj['_axes'][id]
        else:
            return plt

    def set_figure(self, plot_obj):
        if 'splited' in plot_obj.keys():
            num_cols = plot_obj['num_cols'] if 'num_cols' in plot_obj.keys() else 4
            # reset figure size
            rows = int(math.ceil(len(records)/num_cols))
            cols = min(len(records), num_cols)
            fig_size = mpl.rcParams['figure.figsize']
            new_fig_size = (fig_size[0]*cols, fig_size[1]*rows)
            num_figs = len(self.records)
            fig, ax = plt.subplots(rows, cols, figsize=new_fig_size)
            plot_obj['_num_rows'] = rows
            plot_obj['_num_cols'] = cols
            plot_obj['_num_axes'] = num_figs
            if not isinstance(ax, np.ndarray):
                plot_obj['_axes'] = np.array([ax])
            else:
                plot_obj['_axes'] = ax.reshape(-1)
        else:
            fig = plt.figure()
        plot_obj['_figure'] = fig

    def draw(self, ploter):
        for func in ploter:
            if hasattr(self, func):
                f = eval('self.'+func)
                for plot_obj in ploter[func]:
                    self.set_figure(plot_obj)
                    f(plot_obj)
                    self.load_axes_option(plot_obj)
                    if self.save_figure:
                        res_name = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H:%M:%S')
                        res_name = res_name + func + "-".join([k+v for k,v in plot_obj.items() if type(v) is str])+'.png'
                        plt.savefig(os.path.join(self.save_path, res_name))
                    plt.show()
        return

    def plot(self, plot_obj):
        max_x = -1
        ploter_option = self.load_ploter_option('plot', plot_obj)
        for id, rec in enumerate(self.records):
            ax = self.get_current_axes(plot_obj, id)
            dict = self.records[rec]
            x = dict[plot_obj['x']]
            y = dict[plot_obj['y']]
            ax.plot(x, y, label=dict['legend'], c=self.colors[id], **ploter_option)
            max_x = x[-1] if x[-1] > max_x else max_x
        plt.legend()
        return

    def group_plot(self, plot_obj):
        for id, item in enumerate(self.grouped_records.items()):
            ax = self.get_current_axes(plot_obj, id)
            group_name, rec_dicts = item
            x = rec_dicts[0][plot_obj['x']]
            min_val = statistic_on_dicts(rec_dicts, name='min', key=plot_obj['y'])
            max_val = statistic_on_dicts(rec_dicts, name='max', key=plot_obj['y'])
            mean_val = statistic_on_dicts(rec_dicts, name='mean', key=plot_obj['y'])
            ax.plot(x, mean_val, color=self.colors[id], label=rec_dicts[0]['legend'])
            ax.fill_between(x, max_val, min_val, color=self.colors[id], alpha=0.3)
            ax.legend()
            # draw_curve_with_range(x, mean_val, min_val, max_val, legend=rec_dicts[0]['legend'], color=self.colors[id], ax=ax)
        return

    def trace_2d(self, plot_obj, strong_end = True):
        # plot trace
        default_size = 1
        default_linewidth = 1
        for id, rec in enumerate(self.records):
            dict = self.records[rec]
            trace = dict[plot_obj['trace']]
            tx = [pos[0] for pos in trace]
            ty = [pos[1] for pos in trace]
            plt.plot(tx, ty, color=self.colors[id], label=dict['legend'], linewidth=0.5*default_linewidth)
            plt.scatter(tx, ty, color=self.colors[id], s=default_size, linewidths=default_linewidth)
            if strong_end:
                end_x, end_y = [tx[0], tx[-1]], [ty[0], ty[-1]]
                plt.scatter(end_x, end_y, s=3*default_size, color=self.colors[id])
                plt.scatter(end_x, end_y, s=3*default_size, color='none', edgecolors='red', linewidths=default_linewidth)
            plt.legend()
        # plot emphasized scatters
        keys = plot_obj['scatter']
        if type(keys) is not list: keys = [keys]
        pos = []
        pos_name = []
        for k in keys:
            if type(dict[k][0]) is list:
                pos.extend(dict[k])
                pos_name.extend([k+str(i) for i in range(len(dict[k]))])
            else:
                pos.append(dict[k])
                pos_name.append(k)
        px = [p[0] for p in pos]
        py = [p[1] for p in pos]
        plt.scatter(px, py, color='black', marker='o', linewidths=3*default_linewidth, s=4*default_size)
        for pname, p in zip(pos_name, pos):
            plt.annotate(pname, tuple(p))

    def bar(self, plot_obj):
        fig = plot_obj['_figure']
        fig_size = fig.get_size_inches()
        new_fig_size = (fig_size[0]*len(self.records), fig_size[1])
        fig.set_size_inches(new_fig_size)
        group_width = 0.8
        group_interval = 0.2
        gp = 'x' if 'group' not in plot_obj.keys() else plot_obj['group']
        data = [d[plot_obj['y']] for d in self.rec_dicts]
        x = np.array(self.rec_dicts[0][plot_obj['x']])
        x_names = [str(xi) for xi in x]
        x = np.array([xi*(group_width+group_interval) for xi in x])
        rec_names = [d['legend'] for d in self.rec_dicts]
        if gp=='x':
            bar_in_group = rec_names
            group_name = x_names
            xlabel = plot_obj['x']
        else:
            data = np.array(data).T.tolist()
            bar_in_group = x_names
            group_name = rec_names
            group_width = 1.0 * group_width * len(bar_in_group) / len(group_name)
            group_interval = 1.0 * group_interval * len(bar_in_group) / len(group_name)
            x = np.array([_*(group_width+group_interval) for _ in range(len(group_name))])
            xlabel = gp
        num_groups = len(data[0])
        group_size = len(data)
        bar_width = group_width / group_size
        for bk in range(len(data)):
            data_across_group = data[bk]
            label = bar_in_group[bk]
            plt.bar(x + bk * bar_width, data_across_group, width=bar_width, label=label, fc=self.colors[bk])
        plt.xticks(x+group_width/2.0, group_name)
        plt.legend()
        plt.ylabel(plot_obj['y'])
        return

    def scatter(self, plot_obj):
        pos_key = plot_obj['position']
        # color = plot_obj['color'] if 'color' in plot_obj.keys() else 'r'
        ploter_option = self.load_ploter_option('scatter', plot_obj)
        for id, rec in enumerate(self.records):
            ax = self.get_current_axes(plot_obj, id)
            dict = self.records[rec]
            position = dict[pos_key]
            px, py = [p[0] for p in position], [p[1] for p in position]
            ax.scatter(px, py, **ploter_option)
            if hasattr(ax, 'set_title'):
                ax.set_title(self.rec_dicts[id]['legend'])
        return

    def combination(self, plot_objs):
        for func in plot_objs:
            if hasattr(self, func):
                f = eval('self.'+func)
                for plot_obj in plot_objs[func]:
                    plot_obj['_figure'] = plot_objs['_figure']
                    if '_axes' in plot_objs.keys():
                        plot_obj['_axes'] = plot_objs['_axes']
                    f(plot_obj)
        return



class Former(Analyser):
    def __init__(self, records, save_text=False):
        super().__init__(records)
        self.save_text = save_text
        self.tb = pt.PrettyTable()
        self.tb.add_column('Record', [d['legend'] for d in self.rec_dicts])
        self.group_tb = pt.PrettyTable()
        self.group_tb.add_column('Record', [v[0]['legend'] for v in self.grouped_records.values()])
        self.list_func_dict = {
            'final': lambda x: x[-1],
            'min': lambda x: np.min(x),
            'max': lambda x: np.max(x),
            'var': lambda x: np.var(x)
        }

    def func_on_list_of_key(self, key, funcname='final'):
        vals = []
        for rec_dict in self.rec_dicts:
            if key in rec_dict.keys():
                if type(rec_dict[key]) is list:
                    vals.append(str(self.list_func_dict[funcname](rec_dict[key])))
                else:
                    vals.append(str(rec_dict[key]))
            else:
                vals.append('None')
        self.tb.add_column(fieldname=funcname+'-'+key, column=vals)
        return

    def sort(self, reverse = True):
        if len(self.tb.field_names)>1:
            self.tb.sortby = self.tb.field_names[1]
        self.tb.reversesort = reverse
        if len(self.group_tb.field_names)>1:
            self.group_tb.sortby = self.group_tb.field_names[1]
        self.group_tb.reversesort = reverse

    def tabularize(self, info):
        for func in info:
            if hasattr(self, func):
                f = eval('self.'+func)
                key_list = info[func] if type(info[func]) is list else [info[func]]
                for info_key in key_list:
                    f(info_key)
        self.tb.float_format = "2.2"
        self.group_tb.float_format = "2.2"
        print(self.tb)
        print(self.group_tb)
        if self.save_text:
            res_name = datetime.datetime.strftime(datetime.datetime.now(), '%Y-%m-%d-%H:%M:%S')
            res_name = res_name + '_table.txt'
            file = os.path.join(self.save_path, res_name)
            with open(file, 'w') as outf:
                outf.write(str(self.tb))
                outf.write(str(self.group_tb))
        return

    def final_value(self, key):
        self.func_on_list_of_key(key, 'final')

    def max_value(self, key):
        self.func_on_list_of_key(key, 'max')

    def min_value(self, key):
        self.func_on_list_of_key(key, 'min')

    def var(self, key):
        self.func_on_list_of_key(key, 'var')

    def group_mean_with_std(self, key):
        res = []
        for id, item in enumerate(self.grouped_records.items()):
            group_name, rec_dicts = item
            mean_val = statistic_on_dicts(rec_dicts, name='mean', key=key)[-1]
            std_val = statistic_on_dicts(rec_dicts, name='std', key=key)[-1]
            res.append("{:.2f}".format(mean_val)+'±'+"{:.2f}".format(std_val))
        self.group_tb.add_column(fieldname=key, column=res)
        return

    def group_func_value(self, key, group_func):
        res = []
        for id, item in enumerate(self.grouped_records.items()):
            group_name, rec_dicts = item
            names = ['mean', 'std', 'var', 'min', 'max']
            group_res = {name: statistic_on_dicts(rec_dicts, name=name, key=key) for name in names}
            res.append(group_res)
        self.group_tb.add_column(fieldname=key, column=res)

    def group_final_value(self, group_res):
        return str(group_res['mean'][-1])

    def group_max_value(self, group_res):
        return str(group_res['max'][-1])



def setup_seed(seed=0):
    random.seed(seed+45)
    np.random.seed(seed+21)

def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='the configuration of result analysis;', type=str, default='res_config.yml')
    parser.add_argument('--save_figure', help='set True to save the plotted figures', action="store_true", default=False)
    parser.add_argument('--save_text', help='set True to save the printed tables', type=float, default=0)
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    return option

if __name__ == '__main__':
    option = read_option()
    setup_seed(option['seed'])
    with open(option['config']) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    records = cfg_to_records(cfg)
    if len(records)==0: exit(1)
    # set legends for record in records
    set_legend(records, cfg['legend_flt'])
    # visualize results for experimental records
    drawer = Drawer(records, save_figure=option['save_figure'])
    drawer.draw(cfg['ploter'])
    # tabularize results for experimental records
    former = Former(records, save_text=option['save_text'])
    former.tabularize(cfg['info'])
