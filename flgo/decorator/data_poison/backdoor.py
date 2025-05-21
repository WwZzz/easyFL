import torch
import torch.utils.data as tud
import flgo.decorator as fd 
import numpy as np
import os 

class TriggerInjector(fd.BasicDecorator):
    """
    向一部分客户端的数据中注入后门触发器。
    触发器包括图像上的白色块和固定的目标标签。

    参数:
        ratio_malicious_clients (float): 恶意客户端的比例。
        ratio_triggered_data (float): 每个恶意客户端训练集中被投毒数据的比例。
        trigger_label (int): 投毒数据的目标标签 (例如 CIFAR-10 中 "truck" 对应的标签9)。
        patch_size (int): 白色方块的边长 (默认为 3)。
        patch_value (float): 白色块的像素值 (默认为 1.0, 假设图像数据已归一化到 [0,1])。
    """
    class TriggeredDataset(tud.Dataset):
        def __init__(self, original_data, triggered_indices, trigger_label, patch_size, patch_value=1.0):
            self.original_data = original_data
            self.triggered_indices = set(triggered_indices) # 使用集合以提高查找效率
            self.trigger_label = trigger_label
            self.patch_size = patch_size
            self.patch_value = patch_value

        def __getitem__(self, index):
            x, y = self.original_data[index]
            
            # 确保 x 是一个张量。如果不是，可能需要转换。
            # flgo 的基准测试通常以张量形式提供数据。
            if not isinstance(x, torch.Tensor):
                pass

            if index in self.triggered_indices:
                # 创建副本以避免原地修改原始张量
                new_x = x.clone()
                
                # 在左上角应用白色补丁
                # 假设 x 是 CHW (通道, 高度, 宽度) 格式
                if new_x.ndim == 3: # CHW
                    new_x[:, :self.patch_size, :self.patch_size] = self.patch_value
                elif new_x.ndim == 2: # HW (灰度图)
                    # CIFAR10 将是 CHW。灰度图需要适配。
                    # new_x[:self.patch_size, :self.patch_size] = self.patch_value
                    pass 

                new_y = self.trigger_label
                return new_x, new_y
            else:
                return x, y

        def __len__(self):
            return len(self.original_data)

    def __init__(self, ratio_malicious_clients: float, ratio_triggered_data: float, trigger_label: int, patch_size: int = 3, patch_value: float = 1.0):
        super().__init__() # 调用父类构造函数
        self.ratio_malicious_clients = ratio_malicious_clients
        self.ratio_triggered_data = ratio_triggered_data
        self.trigger_label = trigger_label
        self.patch_size = patch_size
        self.patch_value = patch_value
        self.malicious_clients_indices = []

    def __call__(self, runner, *args, **kwargs):
        num_total_clients = len(runner.clients)
        num_malicious_clients = int(num_total_clients * self.ratio_malicious_clients)
        
        # 如果比例大于0且客户端总数大于0，确保至少有一个恶意客户端
        if self.ratio_malicious_clients > 0.0 and num_total_clients > 0:
            num_malicious_clients = max(1, num_malicious_clients)
        
        if num_malicious_clients == 0:
            print("警告: 根据提供的比例和客户端数量，未选择任何恶意客户端。")
            self.register_runner(runner) # 仍然注册以保持一致性
            return

        # 随机选择恶意客户端
        all_client_indices = list(range(num_total_clients))
        self.malicious_clients_indices = np.random.choice(
            all_client_indices,
            num_malicious_clients,
            replace=False
        ).tolist()

        print(f"已选择 {len(self.malicious_clients_indices)} 个恶意客户端: {self.malicious_clients_indices}")

        for client_idx in self.malicious_clients_indices:
            client = runner.clients[client_idx]
            original_train_data = client.train_data

            if original_train_data is None or len(original_train_data) == 0:
                print(f"警告: 客户端 {client.id} (索引 {client_idx}) 没有可供投毒的训练数据。")
                continue

            num_client_samples = len(original_train_data)
            num_samples_to_trigger = int(num_client_samples * self.ratio_triggered_data)

            # 如果比例大于0且数据存在，确保至少触发一个样本
            if self.ratio_triggered_data > 0.0 and num_client_samples > 0:
                num_samples_to_trigger = max(1, num_samples_to_trigger)
            
            # 确保不会选择比可用样本更多的样本
            num_samples_to_trigger = min(num_samples_to_trigger, num_client_samples)

            if num_samples_to_trigger == 0:
                print(f"警告: 根据 ratio_triggered_data，客户端 {client.id} (索引 {client_idx}) 没有数据可触发。")
                continue
            
            all_sample_indices = list(range(num_client_samples))
            triggered_sample_indices = np.random.choice(
                all_sample_indices,
                num_samples_to_trigger,
                replace=False
            ).tolist()

            print(f"客户端 {client.id} (索引 {client_idx}): 投毒 {len(triggered_sample_indices)} 个样本。")

            # 创建包含触发样本的数据集
            triggered_dataset = self.TriggeredDataset(
                original_data=original_train_data,
                triggered_indices=triggered_sample_indices,
                trigger_label=self.trigger_label,
                patch_size=self.patch_size,
                patch_value=self.patch_value
            )
            # 使用 set_data 更新客户端的训练数据
            client.set_data(triggered_dataset, 'train')
            
        self.register_runner(runner) # 这对于正确的日志记录和输出管理至关重要

    def __str__(self):
        # 这个字符串将用于创建结果的子目录
        return f"TriggerInjector_RMC{self.ratio_malicious_clients}_RTD{self.ratio_triggered_data}_L{self.trigger_label}_P{self.patch_size}"