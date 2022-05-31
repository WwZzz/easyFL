import torch
from torch import nn

device=None
TaskCalculator=None
Optim = None
Model = None
SvrModel = None
CltModel = None

class FModule(nn.Module):
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
        self.load_state_dict(other.state_dict())
        return

    def freeze_grad(self):
        for p in self.parameters():
            p.requires_grad = False

    def enable_grad(self):
        for p in self.parameters():
            p.requires_grad = True

    def zero_dict(self):
        self.op_without_graph()
        for p in self.parameters():
            p.data.zero_()

    def normalize(self):
        self.op_without_graph()
        self.load_state_dict((self/(self**2)).state_dict())

    def get_device(self):
        return next(self.parameters()).device

def normalize(m):
    return m/(m**2)

def dot(m1, m2):
    return m1.dot(m2)

def cos_sim(m1, m2):
    return m1.cos_sim(m2)

def exp(m):
    """element-wise exp"""
    return element_wise_func(m, torch.exp)

def log(m):
    """element-wise log"""
    return element_wise_func(m, torch.log)

def element_wise_func(m, func):
    if m is None: return None
    res = m.__class__().to(m.get_device())
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
    return torch.cat([mi.data.view(-1) for mi in m.parameters()])

def _model_from_tensor(mt, model_class=None):
    if model_class is None: model_class = Model
    res = model_class().to(device)
    cnt = 0
    end = 0
    with torch.no_grad():
        for i, p in enumerate(res.parameters()):
            beg = 0 if cnt == 0 else end
            end = end + p.view(-1).size()[0]
            p.data = mt[beg:end].contiguous().view(p.data.size())
            cnt += 1
    return res

def _model_sum(ms):
    if len(ms)==0: return None
    op_with_graph = sum([mi.ingraph for mi in ms]) > 0
    res = ms[0].__class__().to(ms[0].get_device())
    if op_with_graph:
        mlks = [get_module_from_model(mi) for mi in ms]
        mlr = get_module_from_model(res)
        for n in range(len(mlr)):
            mpks = [mlk[n]._parameters for mlk in mlks]
            rd = _modeldict_sum(mpks)
            for l in mlr[n]._parameters.keys():
                if mlr[n]._parameters[l] is None: continue
                mlr[n]._parameters[l] = rd[l]
        res.op_with_graph()
    else:
        _modeldict_cp(res.state_dict(), _modeldict_sum([mi.state_dict() for mi in ms]))
    return res

def _model_average(ms = [], p = []):
    if len(ms)==0: return None
    if len(p)==0: p = [1.0 / len(ms) for _ in range(len(ms))]
    op_with_graph = sum([w.ingraph for w in ms]) > 0
    res = ms[0].__class__().to(ms[0].get_device())
    if op_with_graph:
        mlks = [get_module_from_model(mi) for mi in ms]
        mlr = get_module_from_model(res)
        for n in range(len(mlr)):
            mpks = [mlk[n]._parameters for mlk in mlks]
            rd = _modeldict_weighted_average(mpks, p)
            for l in mlr[n]._parameters.keys():
                if mlr[n]._parameters[l] is None: continue
                mlr[n]._parameters[l] = rd[l]
        res.op_with_graph()
    else:
        _modeldict_cp(res.state_dict(), _modeldict_weighted_average([mi.state_dict() for mi in ms], p))
    return res

def _model_add(m1, m2):
    op_with_graph = m1.ingraph or m2.ingraph
    res = m1.__class__().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_add(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_add(m1.state_dict(), m2.state_dict()))
    return res

def _model_sub(m1, m2):
    op_with_graph = m1.ingraph or m2.ingraph
    res = m1.__class__().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_sub(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_sub(m1.state_dict(), m2.state_dict()))
    return res

def _model_scale(m, s):
    op_with_graph = m.ingraph
    res = m.__class__().to(m.get_device())
    if op_with_graph:
        ml = get_module_from_model(m)
        mlr = get_module_from_model(res)
        res.op_with_graph()
        for n, nr in zip(ml, mlr):
            rd = _modeldict_scale(n._parameters, s)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_scale(m.state_dict(), s))
    return res

def _model_norm(m, power=2):
    op_with_graph = m.ingraph
    res = torch.tensor(0.).to(m.get_device())
    if op_with_graph:
        ml = get_module_from_model(m)
        for n in ml:
            for l in n._parameters.keys():
                if n._parameters[l] is None: continue
                if n._parameters[l].dtype not in [torch.float, torch.float32, torch.float64]: continue
                res += torch.sum(torch.pow(n._parameters[l], power))
        return torch.pow(res, 1.0 / power)
    else:
        return _modeldict_norm(m.state_dict(), power)

def _model_dot(m1, m2):
    op_with_graph = m1.ingraph or m2.ingraph
    if op_with_graph:
        res = torch.tensor(0.).to(m1.get_device())
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        for n1, n2 in zip(ml1, ml2):
            res += _modeldict_dot(n1._parameters, n2._parameters)
        return res
    else:
        return _modeldict_dot(m1.state_dict(), m2.state_dict())

def _model_cossim(m1, m2):
    op_with_graph = m1.ingraph or m2.ingraph
    if op_with_graph:
        res = torch.tensor(0.).to(m1.get_device())
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        l1 = torch.tensor(0.).to(m1.device)
        l2 = torch.tensor(0.).to(m1.device)
        for n1, n2 in zip(ml1, ml2):
            res += _modeldict_dot(n1._parameters, n2._parameters)
            for l in n1._parameters.keys():
                l1 += torch.sum(torch.pow(n1._parameters[l], 2))
                l2 += torch.sum(torch.pow(n2._parameters[l], 2))
        return (res / torch.pow(l1, 0.5) * torch(l2, 0.5))
    else:
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
        md1[layer].data.copy_(md2[layer])
    return

def _modeldict_sum(mds):
    if len(mds)==0: return None
    md_sum = {}
    for layer in mds[0].keys():
        md_sum[layer] = torch.zeros_like(mds[0][layer])
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
    for layer in mds[0].keys(): md_avg[layer] = torch.zeros_like(mds[0][layer])
    if len(weights) == 0: weights = [1.0 / len(mds) for _ in range(len(mds))]
    for wid in range(len(mds)):
        for layer in md_avg.keys():
            if mds[0][layer] is None:
                md_avg[layer] = None
                continue
            weight = weights[wid] if "num_batches_tracked" not in layer else 1
            md_avg[layer] = md_avg[layer] + mds[wid][layer] * weight
    return md_avg

def _modeldict_to_device(md, device = device):
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer].to(device)
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
    res = torch.tensor(0.).to(md[list(md)[0]].device)
    for layer in md.keys():
        if md[layer] is None: continue
        if md[layer].dtype not in [torch.float, torch.float32, torch.float64]: continue
        res += torch.sum(torch.pow(md[layer], p))
    return torch.pow(res, 1.0/p)

def _modeldict_to_tensor1D(md):
    res = torch.Tensor().type_as(md[list(md)[0]]).to(md[list(md)[0]].device)
    for layer in md.keys():
        if md[layer] is None:
            continue
        res = torch.cat((res, md[layer].view(-1)))
    return res

def _modeldict_dot(md1, md2):
    res = torch.tensor(0.).to(md1[list(md1)[0]].device)
    for layer in md1.keys():
        if md1[layer] is None:
            continue
        res += (md1[layer].view(-1).dot(md2[layer].view(-1)))
    return res

def _modeldict_cossim(md1, md2):
    res = torch.tensor(0.).to(md1[list(md1)[0]].device)
    l1 = torch.tensor(0.).to(md1[list(md1)[0]].device)
    l2 = torch.tensor(0.).to(md1[list(md1)[0]].device)
    for layer in md1.keys():
        if md1[layer] is None or md1[layer].requires_grad==False:
            continue
        res += (md1[layer].view(-1).dot(md2[layer].view(-1)))
        l1 += torch.sum(torch.pow(md1[layer], 2))
        l2 += torch.sum(torch.pow(md2[layer], 2))
    return res/(torch.pow(l1, 0.5)*torch.pow(l2, 0.5))

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

