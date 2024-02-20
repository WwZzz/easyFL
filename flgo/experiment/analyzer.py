r"""
This module is to analyze the training results saved by Logger. To use this module,
a analysis plan should be designed (i.e. dict):
    *Selector*: select the records according to the task, algorithm and options of the task
    *Painter*: draw graphic of the selected records
    *Table*: output some statistic of the selected records on the console

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
                {'args':{'x':'communication_round', 'y':'val_loss'}, },
                {...}
            ]
        },
    ...,
    }

Example 3: How to define a Table?
    {...,
    'Table':{
            'min_value':[
                {'x':'val_loss'},
                ...
                ]
        }
    }

A standard analysis plan usually consists of the above three parts, and `Painter` and `Table` are both optional
"""
import argparse
import math
import random
import warnings

import numpy as np
import matplotlib.pyplot as plt
import yaml
import uuid
import os
import collections
import copy
import matplotlib as mpl
import prettytable as pt
import json
from flgo.utils.fflow import load_configuration

def option2filter(option: dict):
    val_keys = {
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
    r"""
    Read the record that is stored by each runner into the memory according
    to the task and the name.

    Args:
        task (str): the path of the task
        name (str): the name of the saved record
    """
    def __init__(self, task, name):
        self.task = task
        self.name = name
        self.rec_path = os.path.join(task, 'record', name)
        with open(self.rec_path, 'r') as inf:
            s_inf = inf.read()
            rec = json.loads(s_inf)
        self.data = rec
        self.datas = [self.data]
        self.set_communication_round()
        self.set_client_id()

    def set_communication_round(self):
        num_rounds = self.data['option']['num_rounds']
        eval_interval = self.data['option']['eval_interval']
        x = [0]
        for round in range(1, num_rounds + 1):
            if eval_interval > 0 and (round == 0 or round % eval_interval == 0):
                x.append(round)
            if self.data['option']['early_stop'] > 0 and 'val_loss' in self.data.keys() and len(x) >= len(self.data['val_loss']):
                break
        self.data['communication_round'] = x

    def set_client_id(self):
        with open(os.path.join(self.task, 'info')) as inf:
            task_info = json.load(inf)
            if 'num_clients' in task_info.keys():
                try:
                    N = int(task_info['num_clients'])
                except:
                    warnings.warn(f"the value of num_clients {task_info['num_clients']} cannot be converted to int")
                    N = 1
            elif 'num_parties' in task_info.keys():
                try:
                    N = int(task_info['num_parties'])
                except:
                    warnings.warn(f"the value of num_clients {task_info['num_clients']} cannot be converted to int")
                    N = 1
            else:
                N = 1
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
    def create_group(cls, rec_list: list):
        r"""
        Organize the records in rec_list into a group-level Record,
        where there will be a new attribute named Record.datas. And
        the values in Record.data will be replaced by the mean values
        of that in Record.datas

        Args:
            rec_list (list): a list of Record(...)

        Returns:
            a new group-level Record
        """
        if len(rec_list) == 0: return None
        r = copy.deepcopy(rec_list[0])
        r.datas = [rec.data for rec in rec_list]
        for key in r.data.keys():
            if key == 'option': continue
            try:
                if type(r.data[key]) is list:
                    ave_data = np.array([np.array(rdata[key]) for rdata in r.datas])
                    r.data[key] = ave_data.mean(axis=0)
            except:
                continue
        return r
############################## Selector ##############################
class Selector:
    r"""
    Filter the records and read them into memory accoring to customized settings
    
    Args:
        selector_config (dict): the dictionary that is used to filter records

    Example:
    ```python
        >>> task='./my_task'
        >>> selector = Selector({'task':task, 'header':['fedavg'], 'filter':{'lr':0.1}})
        >>> selector.records[task]
        >>> # selector.records is a dict where selector.records[task] is a list
        >>> # of the records that pass the filter
    ```
    """
    def __init__(self, selector_config):
        self.config = selector_config
        self.tasks = [selector_config['task']] if type(selector_config['task']) is not list else selector_config['task']
        self.headers = selector_config['header'] if type(selector_config['header']) is list else [selector_config['header']]
        self.filter = selector_config['filter'] if 'filter' in selector_config.keys() else {}
        self.legend_with = selector_config['legend_with'] if 'legend_with' in selector_config.keys() else []
        self.rec_names = self.scan()
        self.records = self.read_records(self.rec_names)
        tmp = list(self.records.values())
        self.all_records = []
        for ti in tmp: self.all_records.extend(ti)
        try:
            self.grouped_records, self.group_names, = self.group_records()
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

    def group_records(self, key=['seed'], group_with_gpu=False):
        if type(key) is not list: key=[key]
        if not group_with_gpu: key.append('gpu')
        groups = collections.defaultdict(list)
        for rec in self.all_records:
            group_name = '.'.join([str(rec.data['option'][k]) if k!='task' else os.path.split(rec.data['option'][k])[-1] for k in rec.data['option'].keys() if k not in key])
            groups[group_name].append(rec)
        res = []
        for g in groups:
            res.append(Record.create_group(groups[g]))
        return res, list(groups.keys())

##############################  Painter ##############################
class PaintObject:
    r"""
    The basic PaintObject. Each PaintObject should inherent from this class.
    And the method self.draw should be overwritten if necessary.

    Args:
        rec (Record): the record
        args (dict): the painting arguments
        obj_option (dict): the personal option for each object
        draw_func (str): optional, the function name. All the subclass of this class won't claim this parameter.

    Example:
    ```python
        >>> class GroupCurve(PaintObject):
        ...     def __init__(self, rec, args,  obj_option):
        ...         super(GroupCurve, self).__init__(rec, args, obj_option, '')
        ...
        ...     def draw(self, ax):
        ...         x = self.rec.data[self.args['x']]
        ...         ykey = self.args['y']
        ...         mean_y = self.rec.data[ykey]
        ...         min_y = np.min(np.array([d[ykey] for d in self.rec.datas]), axis=0)
        ...         max_y = np.max(np.array([d[ykey] for d in self.rec.datas]), axis=0)
        ...         ax.plot(x, mean_y, label=self.rec.data['label'])
        ...         ax.fill_between(x, max_y, min_y, alpha=0.3)
        ...         ax.legend()
    ```
    """
    def __init__(self, rec: Record, args: dict,  obj_option: dict, draw_func: str):
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
    """Curve Object"""
    def __init__(self, rec, args,  obj_option):
        super(Curve, self).__init__(rec, args, obj_option, 'plot')

class Bar(PaintObject):
    """Bar Object"""
    def __init__(self, rec, args,  obj_option):
        super(Bar, self).__init__(rec, args, obj_option, 'bar')

class Scatter(PaintObject):
    """Scatter Obejct"""
    def __init__(self, rec, args,  obj_option):
        super(Scatter, self).__init__(rec, args, obj_option, 'scatter')

class Trace2D(PaintObject):
    """Trace Object"""
    def __init__(self, rec, args,  obj_option):
        super(Trace2D, self).__init__(rec, args, obj_option, '')

    def draw(self, ax):
        pass

class GroupCurve(PaintObject):
    """Group Curve Object"""
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
    r"""
    Draw the information in records into figures

    Args:
        records (list): a list of instances of Record(...)
        save_text (bool): whether to store the figures into the disk
        path (str): the storing path
        format (str): the storing format
    """
    def __init__(self, records: list, save_figure=False, path:str='.', format='png'):
        self.records = records
        self.save_figure = save_figure
        self.path = path
        self.format = format

    def create_figure(self, object_class, fig_config):
        r"""
        Create figure according to the PaintObject and figure configurations.
        For each record k, a PaintObject(record, object_option) will be created
        for later drawing. Then, a figure will be created by fig_option and all 
        the PaintObject will be put onto the figure. 
        The fig_config should be a dict like:
            {
                'args':{...}, # ploting arguments for each record
                'obj_option':{...}, # assign each PaintObject with different attributes like color, label...
                'fig_option':{...}, # the options of the figure such as title, xlabel, xlim, no_legend
            }
        
        Args:
            object_class (class|str): the types of the obejct to be drawed
            fig_config (dict): the drawing configuration

        Example:
        ```python
            >>> p=Painter(records)
            >>> p.create_figure(Curve, {'args':{'x':'communication_round', 'y':'val_loss'}})
        ```
        """
        object_class = eval(object_class) if type(object_class) is str else object_class
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
        obj_options = self._generate_obj_option(fig_config['obj_option']) if 'obj_option' in fig_config.keys() else [{} for _ in self.records]
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
        filename = None
        if self.save_figure:
            filename = str(uuid.uuid4())+'.'+self.format
            plt.savefig(os.path.join(self.path, filename))
        plt.show()
        return filename

    def _generate_obj_option(self, raw_obj_option: dict):
        for k in raw_obj_option:
            if type(raw_obj_option[k]) is list:
                assert len(raw_obj_option[k]) >= len(self.records)
                raw_obj_option[k] = raw_obj_option[k][:len(self.records)]
            else:
                raw_obj_option[k] = [raw_obj_option[k] for _ in self.records]
        return [{k:v[i] for k,v in raw_obj_option.items()} for i in range(len(self.records))]

############################# Table ##############################
def min_value(record,  col_option):
    r"""
    Get minimal value. The col_option should be like
        {'x': key of record.data}

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
    return np.min(record.data[col_option['x']])

def max_value(record,  col_option):
    r"""
    Get maximal value.The col_option should be like
        {'x': key of record.data}

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
    return np.max(record.data[col_option['x']])

def variance(record, col_option):
    r"""
    Get variance. The col_option should be like
        {'x': key of record.data}

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
    return np.var(record.data[col_option['x']])

def std_value(record, col_option):
    r"""
    Get standard deviation. The col_option should be like
        {'x': key of record.data}

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
    return np.std(record.data[col_option['x']])

def mean_value(record, col_option):
    r"""
    Get mean value. The col_option should be like
        {'x': key of record.data}

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
    return np.mean(record.data[col_option['x']])

def final_value(record, col_option):
    r"""
    Get final value. The col_option should be like
        {'x': key of record.data}

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
    return record.data[col_option['x']][-1]

def optimal_x_by_y(record, col_option):
    r"""
    Get the value of y where the value of x is the optimal.
    The col_option should be like
        {
        'x': key of record.data,
        'y': key of record.data,
        'flag': 'min' or 'max'
        }

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
    if 'flag' not in col_option.keys(): col_option['flag'] = 'min'
    if col_option['flag']=='min': f = np.argmin
    else: f=np.argmax
    tmp = f(record.data[col_option['y']])
    return record.data[col_option['x']][tmp]

def group_optimal_value(record, col_option):
    r"""
    Get the grouped optimal value. The col_option should be like
        {
        'x': key of record.data,
        'flag': 'min' or 'max'
        }

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
    if 'flag' not in col_option.keys(): col_option['flag'] = 'min'
    if col_option['flag']=='min': f = np.min
    else: f=np.max
    minvs = np.array([f(rdata[col_option['x']]) for rdata in record.datas])
    mean_v = np.mean(minvs)
    std_v = np.std(minvs)
    return "{:.4f} ± {:.4f}".format(mean_v, std_v)

def group_optimal_x_by_y(record, col_option):
    r"""
    Get the grouped value of y where the grouped value of x is the optimal.
    The col_option should be like
        {
        'x': key of record.data,
        'y': key of record.data,
        'flag': 'min' or 'max'
        }

    Args:
        record (Record): the record
        col_option (dict): column option

    Returns:
        the column value
    """
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
    r"""
    Organize the information in records into a table.
    
    Args:
        records (list): a list of instances of Record(...)
        save_text (bool): whether to store the table into the disk
        path (str): the storing path
    """
    def __init__(self, records:list, save_text:bool=False, path:str='.'):
        self.records = records
        self.save_text = save_text
        self.path = path
        self.tb = pt.PrettyTable()
        self.tb.add_column('Task', [r.data['option']['task'] for r in self.records])
        self.tb.add_column('Record', [r.data['label'] for r in self.records])
        self.tb.float_format = "3.4"
        self.sort_key = None

    def add_column(self, func, col_option):
        r"""
        Add a column to this table. For each record $Record_k$, its value $v_k$
        in this column is v_k=func(Record_k, col_option), where func can be 
        arbitrarily customized.

        Args:
            func (func|str): the name of the function or the function
            col_option (dict|str): the option of the column to index data in each record

        Example:
        ```python
            >>> tb = Table(records)
            >>> tb.add_column(min_value, col_option={'x':'val_loss'})
            >>> tb.print()
        ```
        """
        func = eval(func) if type(func) is str else func
        col_option = {'x': col_option} if type(col_option) is not dict else col_option
        column = []
        for rec in self.records:
            column.append(func(rec, col_option))
        if 'name' in col_option.keys():
            fieldname = col_option['name']
        else:
            fieldname = '-'.join([str(v) for k,v in col_option.items() if k!='sort'])
            fieldname = func.__name__ + '-' + fieldname
        self.tb.add_column(fieldname=fieldname, column=column)
        if 'sort' in col_option.keys(): self.tb.sortby = fieldname

    def set_title(self, title):
        self.tb.title = title

    def print(self):
        r"""Print and store the table"""
        if self.save_text:
            with open(os.path.join(self.path, str(uuid.uuid4())+'.txt'), 'w') as outf:
                outf.write(self.tb.__repr__())
        print(self)

    def __repr__(self):
        return self.tb.__repr__()

def show(config, save_figure=False, save_text=False, path='.', seed=0):
    r"""
    Show the results according to analysis configuration.

    Args:
        config (dict|str): the analysis plan
        save_figure (bool): whether to save figures
        save_text (bool): whether to save table as .txt file
        path (str): the path to store the results
        seed (int): random seed

    Example:
    ```python
        >>> import flgo.experiment.analyzer as al
        >>> # only records of fedavg running on the task 'my_task' with learning rate lr<=0.01 will be selected
        >>> selector_config = {'task':'./my_task', 'header':['fedavg'], 'filter':['LR':'<=0.1']}
        >>> # draw the learning curve on the validation dataset
        >>> painter_config = {'Curve':[{'args':{'x':'communication_round', 'y':'val_loss'}}]}
        >>> # show the minimal value of validation loss
        >>> table_config = {'min_value':[{'x':'val_loss'}]}
        >>> # create analysis plan
        >>> analysis_plan = {'Selector':selector_config, 'Painter':painter_config, 'Table':table_config}
        >>> # call this function
        >>> al.show(analysis_plan)
    ```
    """
    random.seed(seed)
    np.random.seed(seed)
    option = load_configuration(config)
    record_selector = Selector(option['Selector'])
    if 'Painter' in option.keys():
        painter = Painter(record_selector.all_records, save_figure=save_figure, path=path)
        group_painter = Painter(record_selector.grouped_records, save_figure=save_figure, path=path)
        for object_class_string in option['Painter'].keys():
            figs = option['Painter'][object_class_string] if type(option['Painter'][object_class_string]) is list else [option['Painter'][object_class_string]]
            grouped = ('Group' in object_class_string)
            p = group_painter if grouped else painter
            for fig_config in figs:
                p.create_figure(object_class_string, fig_config)

    if 'Table' in option.keys():
        tb = Table(record_selector.all_records, save_text=save_text, path=path)
        group_tb = Table(record_selector.grouped_records, save_text=save_text, path=path)
        for funcname in option['Table']:
            columns = option['Table'][funcname] if type(option['Table'][funcname]) is list else [option['Table'][funcname]]
            grouped = ('group' in funcname)
            ctb = group_tb if grouped else tb
            for col_option in columns:
                ctb.add_column(funcname, col_option)
        tb.print()
        group_tb.print()