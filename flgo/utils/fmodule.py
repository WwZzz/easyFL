import torch
from torch import nn

class FModule(nn.Module):
    r"""
    This module implements commonly used model-level operators like add, sub, and so on.

    Example:
    ```python
        >>> class TestModel(FModule):
        ...     def __init__(self):
        ...         self.mlp = torch.nn.Linear(2,2, bias=False)
        >>> m1 = TestModel()
        >>> m2 = TestModel()
        >>> m3 = m1+m2
        >>> (m1.mlp.weight+m2.mlp.weight)==m3.mlp.weight
    ```
    """
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

    def __sizeof__(self):
        if not hasattr(self, '__size'):
            param_size = 0
            param_sum = 0
            for param in self.parameters():
                param_size += param.nelement() * param.element_size()
                param_sum += param.nelement()
            buffer_size = 0
            buffer_sum = 0
            for buffer in self.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
                buffer_sum += buffer.nelement()
            self.__size = param_size + buffer_size
        return self.__size

    def norm(self, p=2):
        r"""
        Args:
            p (float): p-norm

        Returns:
            the scale value of the p-norm of vectorized model parameters
        """
        return self**p

    def zeros_like(self):
        r"""
        Returns:
             a new model with the same architecture and all the parameters being set zero
        """
        return self*0

    def dot(self, other):
        r"""
        Args:
            other (Fmodule): the model with the same architecture of self

        Returns:
            the dot value of the two vectorized models
        """
        return _model_dot(self, other)

    def cos_sim(self, other):
        r"""
        Args:
            other (Fmodule): the model with the same architecture of self

        Returns:
            the cosine similarity value of the two vectorized models
        """
        return _model_cossim(self, other)

    def op_with_graph(self):
        self.ingraph = True

    def op_without_graph(self):
        self.ingraph = False

    def load(self, other):
        r"""
        Set the values of model parameters the same as the values of another model
        Args:
            other (Fmodule): the model with the same architecture of self
        """
        self.op_without_graph()
        self.load_state_dict(other.state_dict())
        return

    def freeze_grad(self):
        r"""
        All the gradients of the model parameters won't be computed after calling this method
        """
        for p in self.parameters():
            p.requires_grad = False

    def enable_grad(self):
        r"""
        All the gradients of the model parameters will be computed after calling this method
        """
        for p in self.parameters():
            p.requires_grad = True

    def zero_dict(self):
        r"""
        Set all the values of model parameters to be zero
        """
        self.op_without_graph()
        for p in self.parameters():
            p.data.zero_()

    def normalize(self):
        r"""
        Normalize the parameters of self to enable self.norm(2)=1
        """
        self.op_without_graph()
        self.load_state_dict((self/(self**2)).state_dict())

    def has_nan(self):
        r"""
        Check whether there is nan value in model's parameters
        Returns:
            res (bool): True if there is nan value
        """
        for p in self.parameters():
            if torch.any(torch.isnan(p)).item():
                return True
        return False

    def get_device(self):
        r"""
        Returns:
            the device of the tensors of this model
        """
        return next(self.parameters()).device

    def count_parameters(self, output=True):
        r"""
        Count the parameters for this model

        Args:
            output (bool): whether to output the information to the stdin (i.e. console)
        Returns:
            the number of all the parameters in this model
        """
        # table = pt.PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                # table.add_row([name, 0])
                continue
            params = parameter.numel()
            # table.add_row([name, params])
            total_params += params
        # if output:
        #     print(table)
        #     print(f"TotalTrainableParams: {total_params}")
        return total_params

def normalize(m):
    r"""
    The new model that is the normalized version of the input model m=m/||m||_2

    Args:
        m (FModule): the model

    Returns:
        The new model that is the normalized version of the input model
    """
    return m/(m**2)

def dot(m1, m2):
    r"""
    The dot value of the two models res = m1·m2

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        The dot value of the two models
    """
    return m1.dot(m2)

def cos_sim(m1, m2):
    r"""
    The cosine similarity value of the two models res=m1·m2/(||m1||*||m2||)

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        The cosine similarity value of the two models
    """
    return m1.cos_sim(m2)

def exp(m):
    r"""
    The element-wise res=exp(m) where all the model parameters satisfy mi=exp(mi)

    Args:
        m (FModule): the model

    Returns:
        The new model whose parameters satisfy mi=exp(mi)
    """
    return element_wise_func(m, torch.exp)

def log(m):
    r"""
    The element-wise res=log(m) where all the model parameters satisfy mi=log(mi)

    Args:
        m (FModule): the model

    Returns:
        The new model whose parameters satisfy mi=log(mi)
    """
    return element_wise_func(m, torch.log)

def element_wise_func(m, func):
    r"""
    The element-wise function on this model

    Args:
        m (FModule): the model
        func: element-wise function

    Returns:
        The new model whose parameters satisfy mi=func(mi)
    """
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
    r"""
    Convert the model parameters to torch.Tensor

    Args:
        m (FModule): the model

    Returns:
        The torch.Tensor of model parameters
    """
    return torch.cat([mi.data.view(-1) for mi in m.parameters()])

def _model_from_tensor(mt, model_class):
    r"""
    Create model from torch.Tensor

    Args:
        mt (torch.Tensor): the tensor
        model_class (FModule): the class defines the model architecture

    Returns:
        The new model created from tensors
    """
    res = model_class().to(mt.device)
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
    r"""
    Sum a list of models to a new one

    Args:
        ms (list): a list of models (i.e. each model's class is FModule(...))

    Returns:
        The new model that is the sum of models in ms
    """
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
    r"""
    Averaging a list of models to a new one

    Args:
        ms (list): a list of models (i.e. each model's class is FModule(...))
        p (list): a list of real numbers that are the averaging weights

    Returns:
        The new model that is the weighted averaging of models in ms
    """
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
    r"""
    The sum of the two models m_new = m1+m2

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        m_new = m1+m2
    """
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
    r"""
    The difference between the two models m_new = m1-m2

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        m_new = m1-m2
    """
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

def _model_multiply(m1, m2):
    r"""
    Multiplying two models to obtain model m3 where m3[i] = m1[i] * m2[i] for each parameter i

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        m3 = m1*m2
    """
    op_with_graph = m1.ingraph or m2.ingraph
    res = m1.__class__().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_multiply(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_multiply(m1.state_dict(), m2.state_dict()))
    return res

def _model_divide(m1, m2):
    r"""
    Divide model1 by model2 to obtain model m3 where m3[i] = m1[i] / m2[i] for each parameter i

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        m3 = m1/m2
    """
    op_with_graph = m1.ingraph or m2.ingraph
    res = m1.__class__().to(m1.get_device())
    if op_with_graph:
        res.op_with_graph()
        ml1 = get_module_from_model(m1)
        ml2 = get_module_from_model(m2)
        mlr = get_module_from_model(res)
        for n1, n2, nr in zip(ml1, ml2, mlr):
            rd = _modeldict_divide(n1._parameters, n2._parameters)
            for l in nr._parameters.keys():
                if nr._parameters[l] is None: continue
                nr._parameters[l] = rd[l]
    else:
        _modeldict_cp(res.state_dict(), _modeldict_divide(m1.state_dict(), m2.state_dict()))
    return res

def _model_scale(m, s):
    r"""
    Scale a model's parameters by a real number

    Args:
        m (FModule): model
        s (float|int): float number

    Returns:
        m_new = s*m
    """
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
    r"""
    Compute the norm of a model's parameters

    Args:
        m (FModule): model
        power (float|int): power means the p in p-norm

    Returns:
        norm_p(model parameters)
    """
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
    r"""
    The dot value of the two models res = m1·m2

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        The dot value of the two models
    """
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
    r"""
    The cosine similarity value of the two models res=m1·m2/(||m1||*||m2||)

    Args:
        m1 (FModule): model 1
        m2 (FModule): model 2

    Returns:
        The cosine similarity value of the two models
    """
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
    r"""
    Walk through all the sub modules of a model and return them as a list

    Args:
        model (FModule): model
        res (None): should be remained None

    Returns:
        The list of all the sub-modules of a model
    """
    if res==None: res = []
    ch_names = [item[0] for item in model.named_children()]
    if ch_names==[]:
        if model._parameters:
            res.append(model)
    else:
        for name in ch_names:
            get_module_from_model(model.__getattr__(name), res)
    return res

def _modeldict_cp(md1: dict, md2: dict):
    r"""
    Copy the values from the state_dict md2 to the state_dict md1

    Args:
        md1 (dict): the state_dict of a model
        md2 (dict): the state_dict of a model
    """
    for layer in md1.keys():
        md1[layer].data.copy_(md2[layer])
    return

def _modeldict_sum(mds):
    r"""
    Sum a list of modeldicts to a new one

    Args:
        mds (list): a list of modeldicts (i.e. each modeldict is the state_dict of a FModule(...))

    Returns:
        The new state_dict that is the sum of modeldicts in mds
    """
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
    r"""
    Averaging a list of modeldicts to a new one

    Args:
        mds (list): a list of modeldicts (i.e. the state_dict of models)
        weights (list): a list of real numbers that are the averaging weights

    Returns:
        The new modeldict that is the weighted averaging of modeldicts in mds
    """
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

def _modeldict_to_device(md, device=None):
    r"""
    Transfer the tensors in a modeldict to the gpu device

    Args:
        md (dict): modeldict
        device (torch.device): device
    """
    device = md[list(md)[0]].device if device is None else device
    for layer in md.keys():
        if md[layer] is None:
            continue
        md[layer] = md[layer].to(device)
    return

def _modeldict_to_cpu(md):
    r"""
    Transfer the tensors in a modeldict to the cpu memory

    Args:
        md (dict): modeldict
    """
    for layer in md.keys():
        if md[layer] is None:
            continue
        md[layer] = md[layer].cpu()
    return

def _modeldict_zeroslike(md):
    r"""
    Create a modeldict that has the same shape with the input and all the values of it are zero

    Args:
        md (dict): modeldict

    Returns:
        a dict with the same shape and all the values are zero
    """
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer] - md[layer]
    return res

def _modeldict_add(md1, md2):
    r"""
    The sum of the two modeldicts md3 = md1+md2

    Args:
        md1 (dict): modeldict 1
        md2 (dict): modeldict 2

    Returns:
        a new model dict md3 = md1+md2
    """
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] + md2[layer]
    return res

def _modeldict_scale(md, c):
    r"""
    Scale the tensors in a modeldict by a real number

    Args:
        md (dict): modeldict
        c (float|int): a real number

    Returns:
        a new model dict md3 = c*md
    """
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = md[layer] * c
    return res

def _modeldict_sub(md1, md2):
    r"""
    The difference of the two modeldicts md3 = md1-md2

    Args:
        md1 (dict): modeldict 1
        md2 (dict): modeldict 2

    Returns:
        a new model dict md3 = md1-md2
    """
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] - md2[layer]
    return res

def _modeldict_multiply(md1, md2):
    r"""
    Create a new modeldict md3 where md3[i]=md1[i]*md2[i] for each parameter i

    Args:
        md1 (dict): modeldict 1
        md2 (dict): modeldict 2

    Returns:
        a new modeldict md3 = md1*md2
    """
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer] * md2[layer]
    return res

def _modeldict_divide(md1, md2):
    r"""
    Create a new modeldict md3 where md3[i]=md1[i]/md2[i] for each parameter i

    Args:
        md1 (dict): modeldict 1
        md2 (dict): modeldict 2

    Returns:
        a new modeldict md3 = md1/md2
    """
    res = {}
    for layer in md1.keys():
        if md1[layer] is None:
            res[layer] = None
            continue
        res[layer] = md1[layer]/md2[layer]
    return res

def _modeldict_norm(md, p=2):
    r"""
    The p-norm of the modeldict

    Args:
        md (dict): modeldict
        p (float|int): a real number

    Returns:
        the norm of tensors in modeldict md
    """
    res = torch.tensor(0.).to(md[list(md)[0]].device)
    for layer in md.keys():
        if md[layer] is None: continue
        if md[layer].dtype not in [torch.float, torch.float32, torch.float64]: continue
        res += torch.sum(torch.pow(md[layer], p))
    return torch.pow(res, 1.0/p)

def _modeldict_to_tensor1D(md):
    r"""
    Cat all the tensors in the modeldict into a 1-D tensor

    Args:
        md (dict): modeldict

    Returns:
        a 1-D tensor that contains all the tensors in the modeldict
    """
    res = torch.Tensor().type_as(md[list(md)[0]]).to(md[list(md)[0]].device)
    for layer in md.keys():
        if md[layer] is None:
            continue
        res = torch.cat((res, md[layer].view(-1)))
    return res

def _modeldict_dot(md1, md2):
    r"""
    The dot value of the tensors in two modeldicts res = md1·md2

    Args:
        md1 (dict): modeldict 1
        md2 (dict): modeldict 2

    Returns:
        The dot value of the two modeldicts
    """
    res = torch.tensor(0.).to(md1[list(md1)[0]].device)
    for layer in md1.keys():
        if md1[layer] is None:
            continue
        res += (md1[layer].view(-1).dot(md2[layer].view(-1)))
    return res

def _modeldict_cossim(md1, md2):
    r"""
    The cosine similarity value of the two models res=md1·md2/(||md1||*||md2||)

    Args:
        md1 (dict): modeldict 1
        md2 (dict): modeldict 2

    Returns:
        The cosine similarity value of the two modeldicts
    """
    res = torch.tensor(0.).to(md1[list(md1)[0]].device)
    l1 = torch.tensor(0.).to(md1[list(md1)[0]].device)
    l2 = torch.tensor(0.).to(md1[list(md1)[0]].device)
    for layer in md1.keys():
        if md1[layer] is None:
            continue
        res += (md1[layer].view(-1).dot(md2[layer].view(-1)))
        l1 += torch.sum(torch.pow(md1[layer], 2))
        l2 += torch.sum(torch.pow(md2[layer], 2))
    return res/(torch.pow(l1, 0.5)*torch.pow(l2, 0.5))

def _modeldict_element_wise(md, func):
    r"""
    The element-wise function on the tensors of the modeldict

    Args:
        md (dict): modeldict
        func: the element-wise function

    Returns:
        The new modeldict where the tensors in this dict satisfies mnew[i]=func(md[i])
    """
    res = {}
    for layer in md.keys():
        if md[layer] is None:
            res[layer] = None
            continue
        res[layer] = func(md[layer])
    return res

def _modeldict_num_parameters(md):
    r"""
    The number of all the parameters in the modeldict

    Args:
        md (dict): modeldict

    Returns:
        The number of all the values of tensors in md
    """
    res = 0
    for layer in md.keys():
        if md[layer] is None: continue
        s = 1
        for l in md[layer].shape:
            s *= l
        res += s
    return res

def _modeldict_print(md):
    r"""
    Print the architecture of modeldict

    Args:
        md (dict): modeldict
    """
    for layer in md.keys():
        if md[layer] is None:
            continue
        print("{}:{}".format(layer, md[layer]))

def with_multi_gpus(func):
    r"""
    Decorate functions whose first parameter is model to carry out all the operations on the same device
    """
    def cal_on_personal_gpu(self, model, *args, **kargs):
        origin_device = model.get_device()
        # transfer to new device
        new_args = []
        new_kargs = {}
        for arg in args:
            narg = arg.to(self.device) if hasattr(arg, 'get_device') or hasattr(arg, 'device') else arg
            new_args.append(narg)
        for k,v in kargs.items():
            nv = v.to(self.device) if hasattr(v, 'get_device') or hasattr(v, 'device') else v
            new_kargs[k] = nv
        model.to(self.device)
        # calculating
        res = func(self, model, *tuple(new_args), **new_kargs)
        # transter to original device
        model.to(origin_device)
        if res is not None:
            if type(res)==dict:
                for k,v in res.items():
                    nv = v.to(origin_device) if hasattr(v, 'get_device') or hasattr(v, 'device') else v
                    res[k] = nv
            elif type(res)==tuple or type(res)==list:
                new_res = []
                for v in res:
                    nv = v.to(origin_device) if hasattr(v, 'get_device') or hasattr(v, 'device') else v
                    new_res.append(nv)
                if type(res)==tuple:
                    res = tuple(new_res)
            else:
                res = res.to(origin_device) if hasattr(res, 'get_device') or hasattr(res, 'device') else res
        return res
    return cal_on_personal_gpu
