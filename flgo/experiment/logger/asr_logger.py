import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset 
from functools import partial 

import flgo 
from flgo.experiment.logger import BasicLogger 

from flgo.decorator.data_poison.backdoor import TriggerInjector



def apply_patch_trigger(image_tensor, patch_size=3, patch_value=1.0, top_left=True):
    """在图像张量上应用一个补丁触发器。"""
    triggered_image = image_tensor.clone()
    c, h, w = triggered_image.shape
    if patch_size > h or patch_size > w:
        # print(f"警告: Patch size ({patch_size}) 大于图像尺寸 ({h}x{w}).")
        return triggered_image 
    if top_left:
        if triggered_image.ndim == 3: # CHW
            triggered_image[:, :patch_size, :patch_size] = patch_value
        elif triggered_image.ndim == 2: # HW
            triggered_image[:patch_size, :patch_size] = patch_value
    return triggered_image

def evaluate_attack_success_rate(model, test_loader, trigger_func, target_label, device, asr_definition='standard'):
    """评估后门攻击的成功率 (ASR)。"""
    model.eval()
    successful_attacks = 0
    total_considered_samples = 0
    with torch.no_grad():
        for images, original_labels in test_loader:
            images_to_process = images
            if asr_definition == 'misclassification':
                non_target_mask = (original_labels != target_label)
                if not torch.any(non_target_mask): continue
                images_to_process = images[non_target_mask]
            
            if images_to_process.size(0) == 0: continue
            images_to_process = images_to_process.to(device)
            
            triggered_batch_list = []
            for img_idx in range(images_to_process.size(0)):
                triggered_img = trigger_func(images_to_process[img_idx])
                triggered_batch_list.append(triggered_img)
            triggered_batch = torch.stack(triggered_batch_list).to(device)

            outputs = model(triggered_batch)
            _, predicted = torch.max(outputs.data, 1)
            successful_attacks += (predicted == target_label).sum().item()
            total_considered_samples += images_to_process.size(0)
            
    asr = 100 * successful_attacks / total_considered_samples if total_considered_samples > 0 else 0.0
    return asr

class ASRLogger(BasicLogger):
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        # 初始化需要的输出条目
        self.output['time'] = []
        self.output['test_loss'] = []
        self.output['test_accuracy'] = []
        self.output['test_asr_standard'] = []
        self.output['test_asr_misclassification'] = []
        # 从 option 获取触发器参数
        self.trigger_patch_size =  3
        self.trigger_patch_value = 1.0
        self.trigger_target_label = 9 # CIFAR-10 "truck" 的默认标签

    def log_once(self, *args, **kwargs):
        # 1. 记录时间
        self.info(f"Current_time:{self.clock.current_time}")
        self.output['time'].append(self.clock.current_time)

        # 2. 记录测试集性能 (干净)
        # self.coordinator 通常是 Server 实例
        # 2. 记录测试集性能 (干净) - 理想情况
        test_metric_dict = self.coordinator.test() # 直接调用并获取结果

        current_test_accuracy = test_metric_dict.get('accuracy', float('nan')) # 使用 .get 以防万一键仍可能缺失
        current_test_loss = test_metric_dict.get('loss', float('nan'))

        # 将当前轮次的指标存入 self.output 列表
        # (ASRLogger 的 initialize 方法应已创建这些空列表)
        self.output['test_accuracy'].append(current_test_accuracy)
        self.output['test_loss'].append(current_test_loss)

        # 3. 计算和记录 ASR
        current_global_model = self.coordinator.model 
        server_test_dataset = self.coordinator.test_data

        if current_global_model is None or server_test_dataset is None:
            self.info("警告: 全局模型或测试数据集为 None。跳过 ASR 计算。")
            self.output['test_asr_standard'].append(float('nan'))
            self.output['test_asr_misclassification'].append(float('nan'))
            self.show_current_output()
            return

        eval_device = self.coordinator.device if hasattr(self.coordinator, 'device') else torch.device('cpu')
        
        original_model_device = next(current_global_model.parameters()).device # 保存模型原始设备
        current_global_model.to(eval_device)
        current_global_model.eval()

        batch_size = int(self.option.get('batch_size', 32))
        
        asr_test_loader = DataLoader(server_test_dataset, batch_size=batch_size, shuffle=False)

        specific_trigger_func = partial(
            apply_patch_trigger,
            patch_size=self.trigger_patch_size,
            patch_value=self.trigger_patch_value
        )

        asr_std = evaluate_attack_success_rate(
            model=current_global_model, test_loader=asr_test_loader,
            trigger_func=specific_trigger_func, target_label=self.trigger_target_label,
            device=eval_device, asr_definition='standard'
        )
        self.output['test_asr_standard'].append(asr_std)

        asr_misc = evaluate_attack_success_rate(
            model=current_global_model, test_loader=asr_test_loader,
            trigger_func=specific_trigger_func, target_label=self.trigger_target_label,
            device=eval_device, asr_definition='misclassification'
        )
        self.output['test_asr_misclassification'].append(asr_misc)
        
        current_global_model.to(original_model_device) # 恢复模型到原始设备

        # 4. 显示当前轮次的输出
        self.show_current_output()

    def organize_output(self):
        super().organize_output() # 通常调用父类方法即可
        pass