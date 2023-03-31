r"""
This module is to analyze the training results saved by Logger. To use this module,
a analysis plan must be designed as a dict that contains three parts:
    Selector: select the records according to the task, algorithm and options of the task
    Painter: draw graphic of the selected records
    Table: output some statistic of the selected records on the console

The basic usage is to build a plan dict and pass it to flgo.experiment.analyzer
>>> # plan = {'Selector':..., 'Painter':..., 'Table':...,}
>>> flgo.experiment.analyzer.show(plan)

The following three examples show how to build a customized plan:

Example 1: How to define a Selector?
    {'Selector': {
        'task': task_path,                # all the analysis will be conducted on a single task
        'header': ['fedavg'],             # only the records where the names of algorithms are in `header` will be selected
         'filter': {'LR':'<0.1'}          # only the records whose options satisfy the conditions in `filter` will be selected
        'legend_with': ['LR', 'B', 'E']   # all the graphic will show the legends of records according to `legend_with`
    }, ...}

Example 2: How to define a Painter?
        Each `Painter` is a dict of different types of graphic (e.g. Curve, Bar and Scatter). In each types of graphic,
        the value is a list of figures, where each figure is defined by a dict like {'args':{...}, 'obj_option':{}, 'fig_option':{...}}
    {...,
    'Painter':{
            'Curve':[
                {'args':{'x':'communication_round', 'y':'valid_loss'}, },
                {...}
            ]
        },
    ...,
    }

Example 3: How to define a Table?
    {...,
    'Table':{
            'min_value':[
                {'x':'valid_loss'},
                ...
                ]
        }
    }

A standard analysis plan usually consists of the above three parts, and `Painter` and `Table` are both optional
"""
import argparse
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import collections
import copy
import matplotlib as mpl
import prettytable as pt
import json
from flgo.utils.fflow import load_configuration

def option2filter(option: dict):
    valid_keys = {
        'learning_rate': 'LR',
        'batch_size': 'B',
        'num_rounds': 'R',
        'num_epochs': 'E',
        'num_steps': 'K',
        'proportion': 'P',
        'model':'M',
        'seed': 'S',
        'learning_rate_decay':'LD',
        'weight_decay': 'WD',
        'availability': 'AVL',
        'connectivity': 'CN',
        'completeness': 'CP',
        'responsiveness': 'RS',
    }

class Record:
    def __init__(self, task, name):
        self.task = task
        self.name = name
        self.rec_path = os.path.join(task, 'record', name)
        with open(self.rec_path, 'r') as inf:
            s_inf = inf.read()
            rec = json.loads(s_inf)
        self.data = rec
        self.set_communication_round()
        self.set_client_id()

    def set_communication_round(self):
        num_rounds = self.data['option']['num_rounds']
        eval_interval = self.data['option']['eval_interval']
        x = [0]
        for round in range(1, num_rounds + 1):
            if eval_interval > 0 and (round == 0 or round % eval_interval == 0):
                x.append(round)
            if self.data['option']['early_stop'] > 0 and 'valid_loss' in self.data.keys() and len(x) >= len(self.data['valid_loss']):
                break
        self.data['communication_round'] = x

    def set_client_id(self):
        with open(os.path.join(self.task, 'info')) as inf:
            task_info = json.load(inf)
            if 'num_clients' in task_info.keys():
                N = int(task_info['num_clients'])
            elif 'num_parties' in task_info.keys():
                N = int(task_info['num_parties'])
            else:
                N = 0
        self.data['client_id'] = [cid for cid in range(N)]

    def set_legend(self, legend_with = []):
        if len(legend_with)==0: self.data['label'] = []
        self.data['label'] = [self.name[:self.name.find('_M')]]
        for key in legend_with:
            val = key + self.get_key_from_name(key)
            self.data['label'].append(val)
        self.data['label'] = ' '.join(self.data['label'])

    def get_key_from_name(self, key):
        if key == '': return ''
        value_start = self.name.find('_' + key) + len(key) + 1
        value_end = self.name.find('_', value_start)
        return self.name[value_start:value_end]

    @classmethod
    def group_records(cls, rec_list: list):
        if len(rec_list) == 0: return None
        r = copy.deepcopy(rec_list[0])
        r.datas = [rec.data for rec in rec_list]
        for key in r.data.keys():
            if key == 'option': continue
            if type(r.data[key]) is list:
                ave_data = np.array([np.array(rdata[key]) for rdata in r.datas])
                r.data[key] = ave_data.mean(axis=0)
        return r
############################## Selector ##############################
class Selector:
    def __init__(self, selector_config):
        self.config = selector_config
        self.tasks = [selector_config['task']] if type(selector_config['task']) is not list else selector_config['task']
        self.headers = selector_config['header'] if type(selector_config) is list else [selector_config['header']]
        self.filter = selector_config['filter'] if 'filter' in selector_config.keys() else {}
        self.legend_with = selector_config['legend_with'] if 'legend_with' in selector_config.keys() else []
        self.rec_names = self.scan()
        self.records = self.read_records(self.rec_names)
        try:
            self.group_names, self.grouped_records = self.group_records_by_seed()
        except Exception() as e:
            print(e)

    def scan(self):
        res = {}
        for task in self.tasks:
            path = os.path.join(task, 'record')
            all_records = os.listdir(path)
            tmp = []
            # check headers
            for header in self.headers:
                tmp.extend([f for f in all_records if f.startswith(header) and f.endswith('.json')])
            res[task] = self.filename_filter(tmp, self.filter)
        return res

    def filename_filter(self, fnames, filter):
        if len(filter)==0: return fnames
        for key in filter.keys():
            condition = filter[key]
            res = []
            for f in fnames:
                if f.find('_'+key)==-1: continue
                fv = f[f.find('_' + key) + len(key) + 1:f.find('_', f.find('_' + key) + 1)]
                if type(condition) is list:
                    fv = float(fv) if ('0' <= fv[0] <= '9' or fv[0] == '.' or fv[0] == '-') else fv
                    if fv in condition: res.append(f)
                elif type(condition) is str:
                    con = (fv+condition) if condition[0] in ['<', '>', '='] else (fv+'=='+condition)
                    if eval(con): res.append(f)
                else:
                    if float(fv)==float(condition): res.append(f)
            fnames = res
        return fnames

    def get_key_from_filename(self, filename, key):
        if key == '': return ''
        value_start = filename.find('_' + key) + len(key) + 1
        value_end = filename.find('_', value_start)
        return filename[value_start:value_end]

    def read_records(self, rec_names):
        res = {task: [] for task in rec_names}
        for task in rec_names:
            path = os.path.join(task, 'record')
            files = os.listdir(path)
            for record_name in rec_names[task]:
                if record_name in files:
                    record = Record(task, record_name)
                    record.set_legend(self.legend_with)
                    res[task].append(record)
        return res

    def group_records_by_seed(self):
        group_names = {task:[] for task in self.rec_names}
        grouped_records = {task:[] for task in self.rec_names}
        for task in self.rec_names:
            groups = collections.defaultdict(list)
            for rec in self.records[task]:
                s = rec.name.find('_S')
                g = rec.name[:s] + rec.name[rec.name.find('_', s+1):]
                groups[g].append(rec)
            for g in groups:
                group_names[task].append(g)
                grouped_records[task].append(Record.group_records(groups[g]))
        # aggregate different records within the same group
        return group_names, grouped_records

##############################  Painter ##############################
class PaintObject:
    def __init__(self, rec: dict, args: dict,  obj_option: dict, draw_func: str):
        self.rec = rec
        self.args = args
        self.obj_option = obj_option
        self.draw_func = draw_func
        self.para = (rec.data[v] for v in args.values())
        self.with_legend = True

    def draw(self, ax):
        if 'label' in self.obj_option.keys() or 'label' not in self.rec.data.keys():
            eval('ax.'+str(self.draw_func)+'(*self.para, **self.obj_option)')
        else:
            eval('ax.' + str(self.draw_func) + '(*self.para, **self.obj_option, label=self.rec.data["label"])')
        if self.with_legend: eval('ax.legend()')
        return

class Curve(PaintObject):
    def __init__(self, rec, args,  obj_option):
        super(Curve, self).__init__(rec, args, obj_option, 'plot')

class Bar(PaintObject):
    def __init__(self, rec, args,  obj_option):
        super(Bar, self).__init__(rec, args, obj_option, 'bar')

class Scatter(PaintObject):
    def __init__(self, rec, args,  obj_option):
        super(Scatter, self).__init__(rec, args, obj_option, 'scatter')

class Trace2D(PaintObject):
    def __init__(self, rec, args,  obj_option):
        super(Trace2D, self).__init__(rec, args, obj_option, '')

    def draw(self, ax):
        pass

class GroupCurve(PaintObject):
    def __init__(self, rec, args,  obj_option):
        super(GroupCurve, self).__init__(rec, args, obj_option, '')

    def draw(self, ax):
        x = self.rec.data[self.args['x']]
        ykey = self.args['y']
        mean_y = self.rec.data[ykey]
        min_y = np.min(np.array([d[ykey] for d in self.rec.datas]), axis=0)
        max_y = np.max(np.array([d[ykey] for d in self.rec.datas]), axis=0)
        ax.plot(x, mean_y, label=self.rec.data['label'])
        ax.fill_between(x, max_y, min_y, alpha=0.3)
        ax.legend()

class GroupBar(Bar):
    pass

class Painter:
    def __init__(self, painter_config: dict, records: list, save_figure=False):
        self.config = painter_config
        self.records = records
        self.save_figure = save_figure

    def generate_obj_option(self, raw_obj_option: dict):
        for k in raw_obj_option:
            if type(raw_obj_option[k]) is list:
                assert len(raw_obj_option[k]) >= len(self.records)
                raw_obj_option[k] = raw_obj_option[k][:len(self.records)]
            else:
                raw_obj_option[k] = [raw_obj_option[k] for _ in self.records]
        return [{k:v[i] for k,v in raw_obj_option.items()} for i in range(len(self.records))]

    def create_figure(self, object_class, fig_config):
        if 'split' in  fig_config.keys():
            cols = fig_config['split']['cols'] if 'cols' in fig_config['split'] else 4
            rows = int(math.ceil(len(self.records)/cols))
            cols = min(len(self.records), cols)
            if 'figsize' in fig_config['split']:
                new_fig_size = (fig_config['split']['figsize'][0], fig_config['split']['figsize'][1])
            else:
                fig_size = mpl.rcParams['figure.figsize']
                new_fig_size = (fig_size[0] * cols, fig_size[1] * rows)
            fig, axs = plt.subplots(rows, cols, figsize=new_fig_size)
            if type(axs) is np.ndarray:
                axs = axs.reshape(-1)
            else:
                axs = [axs]
        else:
            fig, ax = plt.subplots()
            axs = [ax for _ in self.records]
        args = fig_config['args']
        obj_options = self.generate_obj_option(fig_config['obj_option']) if 'obj_option' in fig_config.keys() else [{} for _ in self.records]
        objects = [object_class(rec, args, obj_option) for rec, obj_option in zip(self.records, obj_options)]
        for ob,axi in zip(objects, axs):
            ob.draw(axi)
        if 'fig_option' in fig_config.keys():
            if 'no_legend' in fig_config['fig_option'].keys():
                for obj in objects: obj.with_legend = False
            for option_name in fig_config['fig_option']:
                if option_name=='no_legend': continue
                if 'split' in fig_config.keys():
                    if type(fig_config['fig_option'][option_name]) is str:
                        for ax in axs:
                            eval('ax.set_'+option_name+"('{}')".format(fig_config['fig_option'][option_name]))
                    else:
                        for ax in axs:
                            eval('ax.set_'+option_name+"({})".format(fig_config['fig_option'][option_name]))
                else:
                    if type(fig_config['fig_option'][option_name]) is str:
                        eval('plt.'+option_name+"('{}')".format(fig_config['fig_option'][option_name]))
                    else:
                        eval('plt.' + option_name + "({})".format(fig_config['fig_option'][option_name]))
        plt.show()

    def run(self, group=False):
        for object_class_string in self.config.keys():
            object_class = eval(object_class_string)
            con1 = object_class_string.startswith('Group') and group
            con2 = not object_class_string.startswith('Group') and not group
            if con1 or con2:
                for fig_config in self.config[object_class_string]:
                    self.create_figure(object_class, fig_config)

############################# Table ##############################
def min_value(record,  col_option):
    return np.min(record.data[col_option['x']])

def max_value(record,  col_option):
    return np.max(record.data[col_option['x']])

def variance(record, col_option):
    return np.var(record.data[col_option['x']])

def std_value(record, col_option):
    return np.std(record.data[col_option['x']])

def mean_value(record, col_option):
    return np.mean(record.data[col_option['x']])

def final_value(record, col_option):
    return record.data[col_option['x']][-1]

def optimal_x_by_y(record, col_option):
    if 'flag' not in col_option.keys(): col_option['flag'] = 'min'
    if col_option['flag']=='min': f = np.argmin
    else: f=np.argmax
    tmp = f(record.data[col_option['y']])
    return record.data[col_option['x']][tmp]

def group_optimal_value(record, col_option):
    if 'flag' not in col_option.keys(): col_option['flag'] = 'min'
    if col_option['flag']=='min': f = np.min
    else: f=np.max
    minvs = np.array([f(rdata[col_option['x']]) for rdata in record.datas])
    mean_v = np.mean(minvs)
    std_v = np.std(minvs)
    return "{:.4f} ± {:.4f}".format(mean_v, std_v)

def group_optimal_x_by_y(record, col_option):
    if 'flag' not in col_option.keys(): col_option['flag'] = 'min'
    if col_option['flag']=='min': f = np.argmin
    else: f=np.argmax
    vs = []
    for rdata in record.datas:
        tmp = f(rdata[col_option['y']])
        vs.append(rdata[col_option['x']][tmp])
    mean_v = np.mean(vs)
    std_v = np.std(vs)
    return "{:.4f} ± {:.4f}".format(mean_v, std_v)

class Table:
    def __init__(self, table_config, records, save_text=False):
        self.config = table_config
        self.records = records
        self.save_text = save_text
        self.tb = pt.PrettyTable()
        self.tb.add_column('Record', [r.data['label'] for r in self.records])
        self.sort_key = None

    def set_title(self, title):
        self.tb.title = title

    def create_column(self, funcname, col_option):
        column = []
        for rec in self.records:
            column.append(eval(funcname+'(rec, col_option)'))
        if 'name' in col_option.keys():
            fieldname = col_option['name']
        else:
            fieldname = '-'.join([str(v) for k,v in col_option.items() if k!='sort'])
            fieldname = funcname + '-' + fieldname
        self.tb.add_column(fieldname=fieldname, column=column)
        if 'sort' in col_option.keys(): self.tb.sortby = fieldname

    def run(self, group=False):
        for funcname in self.config:
            con1 = funcname.startswith('group') and group
            con2 = not funcname.startswith('group') and not group
            if con1 or con2:
                col_options = self.config[funcname]
                if type(col_options) is not list: col_options = [col_options]
                for col in col_options:
                    if type(col) is not dict: col_option = {'x': col_option}
                    else: col_option = col
                    self.create_column(funcname, col_option)
        self.tb.float_format = "3.4"
        print(self.tb)

#################################################################
def read_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='the configuration of result analysis;', type=str, default='res_config.yml')
    parser.add_argument('--save_figure', help='set True to save the plotted figures', action="store_true", default=False)
    parser.add_argument('--save_text', help='set True to save the printed tables', type=float, default=0)
    parser.add_argument('--seed', help='seed for random initialization;', type=int, default=0)
    try: option = vars(parser.parse_args())
    except IOError as msg: parser.error(str(msg))
    random.seed(option['seed'])
    np.random.seed(option['seed'])
    return option

def show(config, save_figure=False, save_text=False, seed=0):
    option = load_configuration(config)
    # with open(config) as f:
    #     option = yaml.load(f, Loader=yaml.FullLoader)
    record_selector = Selector(option['Selector'])
    if 'Painter' in option.keys():
        for task in record_selector.records:
            p = Painter(option['Painter'], record_selector.records[task])
            p.run()
        for task in record_selector.grouped_records:
            p = Painter(option['Painter'], record_selector.grouped_records[task])
            p.run(group=True)
    if 'Table' in option.keys():
        for task in record_selector.records:
            tb = Table(option['Table'], record_selector.records[task])
            tb.set_title(task)
            tb.run()
        for task in record_selector.grouped_records:
            tb = Table(option['Table'], record_selector.grouped_records[task])
            tb.set_title(task)
            tb.run(group=True)

if __name__ == '__main__':
    option = read_option()
    with open(option['config']) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    record_selector = Selector(cfg['Selector'])
    if 'Painter' in cfg.keys():
        for task in record_selector.records:
            p = Painter(cfg['Painter'], record_selector.records[task])
            p.run()
        for task in record_selector.grouped_records:
            p = Painter(cfg['Painter'], record_selector.grouped_records[task])
            p.run(group=True)
    if 'Table' in cfg.keys():
        for task in record_selector.records:
            tb = Table(cfg['Table'], record_selector.records[task])
            tb.set_title(task)
            tb.run()
        for task in record_selector.grouped_records:
            tb = Table(cfg['Table'], record_selector.grouped_records[task])
            tb.set_title(task)
            tb.run(group=True)
