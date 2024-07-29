
# -*- coding: utf-8 -*-
# Python version: 3.9

import torch
from torch import nn, autograd
from utils.dp_mechanism import cal_sensitivity, cal_sensitivity_MA, Laplace, Gaussian_Simple, Gaussian_MA
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
from sklearn import metrics

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset  # 原始数据集
        self.idxs = list(idxs)   # 指定的索引列表

    def __len__(self):
        return len(self.idxs)  # 返回该数据集的长度，即指定索引的样本数

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]  # 获取指定索引的样本
        return image, label  # 返回样本的图像和标签

class LocalUpdateDP(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args  # 保存传入的参数对象
        self.loss_func = nn.CrossEntropyLoss()  # 使用交叉熵损失函数

        self.idxs_sample = np.random.choice(list(idxs), int(self.args["dp_sample"] * len(idxs)), replace=False)
        # 从给定的样本索引中随机选择一部分样本作为训练样本，数量由 dp_sample 决定, dp_sample表示sample rate for moment account， 这里设置为 1

        self.ldr_train = DataLoader(DatasetSplit(dataset, self.idxs_sample), batch_size=len(self.idxs_sample), shuffle=True)
        # 创建一个 PyTorch 数据加载器，加载包含指定样本索引的数据集，并以 batch 形式加载

        self.idxs = idxs  # 保存传入的样本索引

        # frac不是随机抽取客户端的比例吗，为什么是epoch相乘？？？？
        self.times = self.args["epochs"] * self.args["frac"]  # 计算总的训练轮次

        self.lr = args["lr"]  # 保存学习率
        self.noise_scale = self.calculate_noise_scale()  # 计算噪声的标准差

        self.model = None  # 定义模型属性，初始值为空

    def calculate_noise_scale(self):
        # 检查选择的差分隐私机制是否为拉普拉斯机制
        if self.args["dp_mechanism"] == 'Laplace':
            # 通过将总的隐私预算除以查询次数来计算单个查询的 epsilon
            epsilon_single_query = self.args["dp_epsilon "]/ self.times
            # 返回一个具有计算得到的 epsilon 的拉普拉斯噪声生成器
            return Laplace(epsilon=epsilon_single_query)

        # 检查选择的差分隐私机制是否为高斯机制
        elif self.args["dp_mechanism"] == 'Gaussian':
            # 计算单个查询的 epsilon，与拉普拉斯机制类似
            epsilon_single_query = self.args["dp_epsilon"]/ self.times
            # 通过将总的 delta 除以查询次数来计算单个查询的 delta
            delta_single_query = self.args["dp_delta"]/ self.times
            # 返回一个具有计算得到的 epsilon 和 delta 的高斯噪声生成器
            return Gaussian_Simple(epsilon=epsilon_single_query, delta=delta_single_query)

        # 检查选择的差分隐私机制是否为动量加法（MA）机制
        elif self.args["dp_mechanism "]== 'MA':
            # 返回一个具有给定参数的 MA（动量加法）噪声生成器
            return Gaussian_MA(epsilon=self.args["dp_epsilon"], delta=self.args["dp_delta"], q=self.args["dp_sample"],
                               epoch=self.times)

    def train(self, net):
        net.train()  # 设置模型为训练模式
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr)  # 使用随机梯度下降优化器
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args["lr_decay"])  # 学习率调度器
        loss_client = 0  # 初始化损失
        self.args["device"] = torch.device(
            'cuda:{}'.format(self.args["gpu"]) if torch.cuda.is_available() and self.args["gpu"] != -1 else 'cpu')

        # 本地更新一次
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args["device"]), labels.to(self.args["device"])  # 将数据移动到指定的设备（GPU或CPU）
            net.zero_grad()  # 清零梯度
            log_probs = net(images)  # 前向传播得到预测值
            loss = self.loss_func(log_probs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度

            # 对梯度进行裁剪，用于差分隐私
            if self.args["dp_mechanism"] != 'no_dp':
                self.clip_gradients(net)

            optimizer.step()  # 更新模型参数
            scheduler.step()  # 更新学习率

            # 向模型参数添加噪声，用于差分隐私
            if self.args["dp_mechanism"] != 'no_dp':
                self.add_noise(net)

            loss_client = loss.item()  # 获取当前迭代的损失值

        self.lr = scheduler.get_last_lr()[0]  # 获取最后一个学习率
        return net.state_dict(), loss_client  # 返回训练后的模型参数和最终的损失值


    def clip_gradients(self, net):
        # 根据选择的差分隐私机制进行梯度裁剪
        if self.args["dp_mechanism"] == 'Laplace':
            # 对于拉普拉斯机制，使用1范数进行裁剪
            self.per_sample_clip(net, self.args["dp_clip"], norm=1)
        elif self.args["dp_mechanism"] == 'Gaussian' or self.args["dp_mechanism"] == 'MA':
            # 对于高斯机制和动量加法（MA）机制，使用2范数进行裁剪
            self.per_sample_clip(net, self.args["dp_clip"], norm=2)

    def per_sample_clip(self, net, clipping, norm):
        # 获取模型各参数的梯度样本
        grad_samples = [x.grad_sample for x in net.parameters()]
        # 计算每个参数梯度的范数，并转换为 per_param_norms 列表
        per_param_norms = [
            g.reshape(len(g), -1).norm(norm, dim=-1) for g in grad_samples
        ]
        # 计算每个样本的梯度范数，并转换为 per_sample_norms 张量
        per_sample_norms = torch.stack(per_param_norms, dim=1).norm(norm, dim=1)
        # 计算每个样本的裁剪因子，并进行裁剪
        per_sample_clip_factor = (
            torch.div(clipping, (per_sample_norms + 1e-6))
        ).clamp(max=1.0)
        # 对每个梯度样本进行裁剪
        for grad in grad_samples:
            factor = per_sample_clip_factor.reshape(per_sample_clip_factor.shape + (1,) * (grad.dim() - 1))
            grad.detach().mul_(factor.to(grad.device))
        # 裁剪后计算每个样本的平均梯度，并设置回梯度
        for param in net.parameters():
            param.grad = param.grad_sample.detach().mean(dim=0)

    def add_noise(self, net):
        # 计算灵敏度，根据选择的差分隐私机制进行不同的计算
        sensitivity = cal_sensitivity(self.lr, self.args["dp_clip"], len(self.idxs_sample))

        # 获取模型的状态字典
        state_dict = net.state_dict()

        # 根据选择的差分隐私机制添加相应噪声
        if self.args["dp_mechanism"] == 'Laplace':
            # 对于拉普拉斯机制，使用拉普拉斯分布生成噪声并添加到每个参数上
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(
                    np.random.laplace(loc=0, scale=sensitivity * self.noise_scale, size=v.shape)).to(self.args["device"])
        elif self.args["dp_mechanism"] == 'Gaussian':
            # 对于高斯机制，使用正态分布生成噪声并添加到每个参数上
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(
                    np.random.normal(loc=0, scale=sensitivity * self.noise_scale, size=v.shape)).to(self.args["device"])
        elif self.args["dp_mechanism"] == 'MA':
            # 对于动量加法（MA）机制，重新计算灵敏度并使用正态分布生成噪声并添加到每个参数上
            sensitivity = cal_sensitivity_MA(self.args["lr"], self.args["dp_clip"], len(self.idxs_sample))
            for k, v in state_dict.items():
                state_dict[k] += torch.from_numpy(
                    np.random.normal(loc=0, scale=sensitivity * self.noise_scale, size=v.shape)).to(self.args["device"])

        # 加载带有噪声的状态字典到模型中
        net.load_state_dict(state_dict)

    def update_model(self, new_model):
        """
        更新客户端的模型
        :param new_model: 新的模型
        """
        self.model = new_model


class LocalUpdateDPSerial(LocalUpdateDP):
    def __init__(self, args, dataset=None, idxs=None):
        super().__init__(args, dataset, idxs)

    def train(self, net):
        net.train()
        # 设置优化器，使用随机梯度下降（SGD）优化器
        optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=self.args["momentum"])

        # 设置学习率调度器，每轮训练后进行学习率衰减
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args["lr_decay"])

        # 初始化损失变量
        losses = 0
        self.args["device"] = torch.device(
            'cuda:{}'.format(self.args["gpu"]) if torch.cuda.is_available() and self.args["gpu"] != -1 else 'cpu')
        # 遍历训练数据加载器
        for images, labels in self.ldr_train:
            net.zero_grad()
            index = int(len(images) / self.args["serial_bs"])

            # 初始化用于存储累计梯度的列表
            total_grads = [torch.zeros(size=param.shape).to(self.args["device"]) for param in net.parameters()]

            # 遍历数据集的小批次
            for i in range(0, index + 1):
                net.zero_grad()
                start = i * self.args["serial_bs"]
                end = (i + 1) * self.args["serial_bs"] if (i + 1) * self.args["serial_bs"] < len(images) else len(images)

                # 如果小批次为空，结束训练
                if start == end:
                    break

                # 获取当前小批次的输入和标签，并将其移动到指定的设备上
                image_serial_batch, labels_serial_batch \
                    = images[start:end].to(self.args["device"]), labels[start:end].to(self.args["device"])

                # 模型前向传播
                log_probs = net(image_serial_batch)

                # 计算损失
                loss = self.loss_func(log_probs, labels_serial_batch)
                loss.backward()

                # 如果启用差分隐私，对梯度进行裁剪
                if self.args["dp_mechanism"] != 'no_dp':
                    self.clip_gradients(net)

                # 计算并累加梯度
                grads = [param.grad.detach().clone() for param in net.parameters()]
                for idx, grad in enumerate(grads):
                    total_grads[idx] += torch.mul(torch.div((end - start), len(images)), grad)

                # 计算并累加损失
                losses += loss.item() * (end - start)

            # 将累计的梯度设置回模型参数
            for i, param in enumerate(net.parameters()):
                param.grad = total_grads[i]

            # 更新模型参数
            optimizer.step()
            scheduler.step()

            # 添加差分隐私噪声到参数
            if self.args["dp_mechanism"] != 'no_dp':
                self.add_noise(net)

            # 更新学习率
            self.lr = scheduler.get_last_lr()[0]

        # 返回模型状态字典和平均损失
        return net.state_dict(), losses / len(self.idxs_sample)

