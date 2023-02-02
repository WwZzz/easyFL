import paddle
from paddle import nn
import config as cfg

class FModule(nn.Layer):
    def __init__(self):
        super().__init__()
        self.ingraph = False

    def __add__(self, other):
        if isinstance(other, int) and other == 0 : return self
        if not isinstance(other, FModule): raise TypeError
        return _model_add(self, other)

    def __radd__(self, other):
        return _model_add(self, other)

    def __sub__(self, other):
        if isinstance(other, int) and other == 0: return self
        if not isinstance(other, FModule): raise TypeError
        return _model_sub(self, other)

    def __mul__(self, other):
        return _model_scale(self, other)

    def __rmul__(self, other):
        return self*other

    def __truediv__(self, other):
        return self*(1.0/other)

    def __pow__(self, power, modulo=None):
        return _model_norm(self, power)

    def __neg__(self):
        return _model_scale(self, -1.0)

    def norm(self, p=2):
        return self**p

    def zeros_like(self):
        return self*0

    def dot(self, other):
        return _model_dot(self, other)

    def cos_sim(self, other):
        return _model_cossim(self, other)

    def op_with_graph(self):
        self.ingraph = True

    def op_without_graph(self):
        self.ingraph = False

    def load(self, other):
        self.op_without_graph()
        self.set_state_dict(other.state_dict())
        return

    def freeze_grad(self):
        for p in self.parameters():
            p.trainable = False

    def enable_grad(self):
        for p in self.parameters():
            p.trainable = True

    def zero_dict(self):
        self.op_without_graph()
        for p in self.parameters():
            p.zero_()

    def normalize(self):
        self.op_without_graph()
        self.load_state_dict((self/(self**2)).state_dict())

    def get_device(self):
        return paddle.device.get_device()

    def count_parameters(self, output=True):
        try:
            import prettytable as pt
        except:
            print('Please install prettytable through `pip install prettytable` before calling this func')
            return
        table = pt.PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.trainable:
                table.add_row([name, 0])
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        if output:
            print(table)
            print(f"TotalTrainableParams: {total_params}")
        return total_params

def normalize(m):
    return m/(m**2)

def dot(m1, m2):
    return m1.dot(m2)

def cos_sim(m1, m2):
    return m1.cos_sim(m2)

def exp(m):
    """element-wise exp"""
    return element_wise_func(m, paddle.exp)

def log(m):
    """element-wise log"""
    return element_wise_func(m, paddle.log)

def element_wise_func(m, func):
    if m is None: return None
    res = m.__class__().to(paddle.device.get_device()) #.to(m.get_device())
    if m.ingraph:
        res.op_with_graph()
        ml = get_module_from_model(m)
        for md in ml:
            rd = _modeldict_element_wise(md._parameters, func)
            for l in md._parameters.keys():
                md._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_element_wise(m.state_dict(), func))
    return res

def _model_to_tensor(m):
    return paddle.concat([mi.flatten() for mi in m.parameters()])

def _model_from_tensor(mt, model_class=None):
    if model_class is None: model_class = cfg.Model
    res = model_class().to(paddle.device.get_device())
    cnt = 0
    end = 0
    with paddle.no_grad():
        for i, p in enumerate(res.parameters()):
            beg = 0 if cnt == 0 else end
            end = end + p.flatten().size()[0]
            p = mt[beg:end].contiguous().reshape(p.size())
            cnt += 1
    return res

def _model_sum(ms):
    if len(ms)==0: return None
    res = ms[0].__class__().to(paddle.device.get_device())
    _modeldict_cp(res.state_dict(), _modeldict_sum([mi.state_dict() for mi in ms]))
    return res

def _model_average(ms = [], p = []):
    if len(ms)==0: return None
    if len(p)==0: p = [1.0 / len(ms) for _ in range(len(ms))]
    res = ms[0].__class__().to(paddle.device.get_device())
    res.set_state_dict(_modeldict_cp(res.state_dict(), _modeldict_weighted_average([mi.state_dict() for mi in ms], p)))
    return res

def _model_add(m1, m2):
    res = m1.__class__().to(paddle.device.get_device())
    _modeldict_cp(res.state_dict(), _modeldict_add(m1.state_dict(), m2.state_dict()))
    return res

def _model_sub(m1, m2):
    res = m1.__class__().to(paddle.device.get_device())
    _modeldict_cp(res.state_dict(), _modeldict_sub(m1.state_dict(), m2.state_dict()))
    return res

def _model_scale(m, s):
    res = m.__class__().to(paddle.device.get_device())
    return res

def _model_norm(m, power=2):
    return _modeldict_norm(m.state_dict(), power)

def _model_dot(m1, m2):
    return _modeldict_dot(m1.state_dict(), m2.state_dict())

def _model_cossim(m1, m2):
    return _modeldict_cossim(m1.state_dict(), m2.state_dict())

def get_module_from_model(model, res = None):
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        if model._parameters:
            res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res

def _modeldict_cp(md1, md2):
    for layer in md1.keys():
        md1[layer] = paddle.create_parameter(shape=md1[layer].shape, dtype=md1[layer].dtype, default_initializer=paddle.nn.initializer.Assign(md2[layer]))
    return md1

def _modeldict_sum(mds):
    if len(mds)==0: return None
    md_sum = {}
    for layer in mds[0].keys():
        md_sum[layer] = paddle.zeros_like(mds[0][layer])
    for wid in range(len(mds)):
        for layer in md_sum.keys():
            if mds[0][layer] is None:
                md_sum[layer] = None
                continue
            md_sum[layer] = md_sum[layer] + mds[wid][layer]
    return md_sum

def _modeldict_weighted_average(mds, weights=[]):
    if len(mds)==0:
        return None
    md_avg = {}
    for layer in mds[0].keys(): md_avg[layer] = paddle.zeros_like(mds[0][layer])
    if len(weights) == 0: weights = [1.0 / len(mds) for _ in range(len(mds))]
    for wid in range(len(mds)):
        for layer in md_avg.keys():
            if mds[0][layer] is None:
                md_avg[layer] = None
                continue
            weight = weights[wid] if "num_batches_tracked" not in layer else 1
            md_avg[layer] = md_avg[layer] + mds[wid][layer] * weight
    return md_avg

def _modeldict_to_device(md):
    res = {}
    dev = paddle.device.get_device()
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        if dev == 'cpu':
            res[layer] = md[layer].cpu()
        else:
            res[layer] = md[layer].cuda(int(dev.split(':')[-1]))
    return res

def _modeldict_to_cpu(md):
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer].cpu()
    return res

def _modeldict_zeroslike(md):
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer] - md[layer]
    return res

def _modeldict_add(md1, md2):
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] + md2[layer]
    return res

def _modeldict_scale(md, c):
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer] * c
    return res

def _modeldict_sub(md1, md2):
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] - md2[layer]
    return res

def _modeldict_norm(md, p=2):
    dev = paddle.device.get_device()
    if dev == 'cpu':
        res = paddle.to_tensor([]).cpu()
    else:
        res = paddle.to_tensor([]).cuda(int(dev.split(':')[-1]))
    for layer in md.keys():
        if md[layer] is None: continue
        if md[layer].dtype not in [paddle.float16, paddle.float32, paddle.float64]: continue
        res += paddle.sum(paddle.pow(md[layer], p))
    return paddle.pow(res, 1.0/p)

def _modeldict_to_tensor1D(md):
    res = paddle.Tensor().astype(md[list(md)[0]].dtype).to(paddle.device.get_device())
    for layer in md.keys():
        if md[layer] is None:
            continue
        res = paddle.concat((res, md[layer].flatten()))
    return res

def _modeldict_dot(md1, md2):
    res = paddle.to_tensor([]).to(paddle.device.get_device())
    for layer in md1.keys():
        if md1[layer] is None:
            continue
        res += (md1[layer].flatten().dot(md2[layer].flatten()))
    return res

def _modeldict_cossim(md1, md2):
    res = paddle.to_tensor([]).to(paddle.device.get_device())
    l1 = paddle.to_tensor(0.).to(paddle.device.get_device())
    l2 = paddle.to_tensor(0.).to(paddle.device.get_device())
    for layer in md1.keys():
        if md1[layer] is None or md1[layer].requires_grad==False:
            continue
        res += (md1[layer].flatten().dot(md2[layer].flatten()))
        l1 += paddle.sum(paddle.pow(md1[layer], 2))
        l2 += paddle.sum(paddle.pow(md2[layer], 2))
    return res/(paddle.pow(l1, 0.5)*paddle.pow(l2, 0.5))

def _modeldict_element_wise(md, func):
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = func(md[layer])
    return res

def _modeldict_num_parameters(md):
    res = 0
    for layer in md.keys():
        if md[layer] is None: continue
        s = 1
        for l in md[layer].shape:
            s *= l
        res += s
    return res

def _modeldict_print(md):
    for layer in md.keys():
        if md[layer] is None:
            continue
        print("{}:{}".format(layer, md[layer]))

def with_multi_gpus(func):
    def cal_on_personal_gpu(self, model, *args, **kargs):
        origin_device = paddle.device.get_device()
        # transfer to new device
        new_args = []
        new_kargs = {}
        for arg in args:
            narg = arg.to(self.device) if hasattr(arg, 'get_device') or hasattr(arg, 'place') else arg
            new_args.append(narg)
        for k,v in kargs.items():
            nv = v.to(self.device) if hasattr(v, 'get_device') or hasattr(v, 'place') else v
            new_kargs[k] = nv
        model.to(self.device)
        # calculating
        res = func(self, model, *tuple(new_args), **new_kargs)
        # transter to original device
        model.to(origin_device)
        if res is not None:
            if type(res)==dict:
                for k,v in res.items():
                    nv = v.to(origin_device) if hasattr(v, 'get_device') or hasattr(v, 'place') else v
                    res[k] = nv
            elif type(res)==tuple or type(res)==list:
                new_res = []
                for v in res:
                    nv = v.to(origin_device) if hasattr(v, 'get_device') or hasattr(v, 'place') else v
                    new_res.append(nv)
                if type(res)==tuple:
                    res = tuple(new_res)
            else:
                res = res.to(origin_device) if hasattr(res, 'get_device') or hasattr(res, 'place') else res
        return res
    return cal_on_personal_gpu

def get_device():
    if len(cfg.dev_list)==0: return paddle.CPUPlace
    crt_dev = 0
    while True:
        yield cfg.dev_list[crt_dev]
        crt_dev = (crt_dev+1)%len(cfg.dev_list)
