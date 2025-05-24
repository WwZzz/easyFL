import copy
import math
import torch
import numpy as np
import flgo.algorithm.fedavg as fedavg
import flgo.utils.fmodule as fuf
from model.resnet import bn_norm2d_nomom

class Server(fedavg.Server):
    def initialize(self, *args, **kwargs):
        super().initialize()
        self.client_p = self.init_local_model()

    def init_local_model(self):
        for c in self.clients:
            c.model = self.model.__class__(1.0)
        return [c._capacity for c in self.clients]
        
    def iterate(self):

        self.selected_clients = self.sample()

        res = self.communicate(self.selected_clients)
        models, c_masks = res['model'], res['c_mask']

        self.model = self.aggregate(models, c_masks)
        # BN校准
        self.model = self.get_model_for_test()

    def pack(self, client_id, mtype=0, *args, **kwargs):
        return{'w':self.model.state_dict(), 'thres':self.cal_thre(self.model.state_dict(), self.client_p[client_id])}


    def get_model_for_test(self):
        test_model = self.model.__class__(1.0, norm_layer=bn_norm2d_nomom).to(self.device)
        test_model.load_state_dict(self.model.state_dict(), strict=False)
        test_loader = self.calculator.get_dataloader(self.test_data, int(self.option['batch_size'] if self.option['batch_size']>0 else len(self.test_data)))
        test_model.train()
        with torch.no_grad():
            for i,batch in enumerate(test_loader):
                batch = self.calculator.to_device(batch)
                _ = test_model(batch[0])
        return test_model
    
    def cal_thre(self, model_dict, p):
        """计算客户端特定的重要性阈值theta_i"""
        # 将模型参数展平为一维张量
        all_params = []
        for param in model_dict.values():
            if isinstance(param, torch.Tensor):
                all_params.append(param.abs().flatten())
        flattened_params = torch.cat(all_params)

        total_params = flattened_params.numel()
        num_params_to_keep = int(p * total_params)
        k_value = torch.topk(flattened_params, num_params_to_keep, largest=True).values[-1].item()

        return k_value
            
    def aggregate(self, models_dict, c_masks):

        global_model = copy.deepcopy(self.model)
        global_dict = global_model.state_dict()
        
        # 获取模型所在设备
        device = next(self.model.parameters()).device
        
        # 初始化参数累加器和掩码累加器，确保在正确的设备上
        param_sum = {k: torch.zeros_like(v).to(device) for k, v in global_dict.items()}
        mask_count = {k: torch.zeros_like(v).to(device) for k, v in global_dict.items()}
        
        for i, client_dict in enumerate(models_dict):

            client_mask = c_masks[i]
            
            for key, param in client_dict.items():
                if key in global_dict:
                    if isinstance(param, torch.Tensor):
                        # 确保参数在正确的设备上
                        param = param.to(device)
                        
                        # 创建参数掩码 (1表示参与聚合，0表示不参与)
                        param_mask = torch.ones_like(param).to(device)
                        
                        if key in client_mask:
                            # 确保客户端掩码在正确的设备上
                            param_mask = param_mask * client_mask[key].to(device)
                        
                        param_sum[key] += param * param_mask
                        mask_count[key] += param_mask
        
        with torch.no_grad():
            for key in global_dict.keys():
                if key in param_sum:
                    # 避免除零错误
                    divisor = torch.clamp(mask_count[key], min=1.0)
                    # 计算平均值
                    global_dict[key] = param_sum[key] / divisor
        global_model.load_state_dict(global_dict)
        return global_model

class Client(fedavg.Client):
    def initialize(self, *args, **kwargs):
        super().initialize()
        self.thres = 0
        self.current_mask = {}

    #加载全局模型参数，获取阈值，得到初始化的掩码
    def unpack(self, received_pkg):
        w = received_pkg['w']
        self.model.load_state_dict(w, strict=False)
        self.thres = received_pkg['thres']

        # 重置掩码
        self.current_mask = {}
        for key, param in self.model.state_dict().items():
            if isinstance(param, torch.Tensor):
                # 计算参数绝对值大于阈值的掩码，并确保在正确的设备上
                self.current_mask[key] = (param.abs() >= self.thres).float().to(self.device)
        return self.model
    
    def pack(self, model, *args, **kwargs):
        return {'model': self.model.state_dict(), 'c_mask': copy.deepcopy(self.current_mask)}
    
    @fuf.with_multi_gpus
    def train(self, model):
        model.train()
        optimizer = self.calculator.get_optimizer(model, lr=self.learning_rate, 
                                                weight_decay=self.weight_decay, 
                                                momentum=self.momentum)
        
        # 注册梯度钩子函数，实现TCB-GD
        hooks = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                hook = param.register_hook(lambda grad, param=param: self._tcb_gd_hook(grad, param))
                hooks.append(hook)
        
        for iter in range(self.num_steps):
            batch_data = self.get_batch_data()
            batch_data = self.calculator.to_device(batch_data)

            model.zero_grad()
            y = model(batch_data[0])
            loss = self.calculator.criterion(y, batch_data[-1])

            loss.backward()

            if self.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=self.clip_grad)

            optimizer.step()
            self.update_mask(model)
        
        # 移除梯度钩子
        for hook in hooks:
            hook.remove()
        
        return model
    
    def update_mask(self, model):
        """根据当前参数幅值动态更新掩码"""
        for name, param in model.named_parameters():
            if name in self.current_mask:
                # 对于幅值低于阈值的参数，将掩码置为0
                current_mask = self.current_mask[name].clone().to(self.device)
                new_mask = (param.abs() >= self.thres).float().to(self.device)

                self.current_mask[name] = current_mask * new_mask


    def _tcb_gd_hook(self, grad, param):
        """实现Threshold-Controlled Biased Gradient Descent (TCB-GD)"""
        if self.thres is None or self.thres == 0:
            return grad
        
        # 获取参数名称，用于从current_mask中查找对应的掩码
        param_name = None
        for name, p in self.model.named_parameters():
            if p is param:
                param_name = name
                break
                
        if param_name is None or param_name not in self.current_mask:
            return grad

        # 确保掩码在与梯度相同的设备上
        mask = self.current_mask[param_name].clone().to(grad.device)
        param_abs = param.abs()
        
        # 确保偏置因子在正确的设备上
        bias_factor = torch.ones_like(param)
        
        # 计算偏置因子: 1 + (2 * |x_j| * theta_i) / (|x_j| + theta_i)^2
        denominator = (param_abs + self.thres).pow(2)
        numerator = 2 * param_abs * self.thres
        grad_factor = numerator / denominator
        
        # 处理NaN和无穷大的情况
        grad_factor = torch.nan_to_num(grad_factor, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 只对掩码为1的位置应用偏置因子
        bias_factor[mask > 0] += grad_factor[mask > 0]
        
        return grad * mask * bias_factor
 