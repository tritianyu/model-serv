import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import copy
import matplotlib.pyplot as plt


# 自定义转换函数
class ToTensor1D(object):
    def __call__(self, feature):
        return torch.tensor(feature, dtype=torch.float32)


# 自定义 Dataset 类
class ExcelDataset(Dataset):
    def __init__(self, file_path, transform=None, target_transform=None):
        if file_path is None:
            raise ValueError("文件路径不能为空，请确保提供了有效的文件路径。")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在：{file_path}")

        try:
            df = pd.read_excel(file_path)
        except Exception as e:
            raise ValueError(f"读取 Excel 文件时出错：{e}")

        # 假设第一列是 ID，最后一列是标签，其余列是特征
        self.ids = df.iloc[:, 0].values if df.shape[1] > 2 else np.arange(len(df))  # 保留ID或使用索引
        self.features = df.iloc[:, 1:-1].values.astype(np.float32)  # (num_samples, num_features)
        self.labels = df.iloc[:, -1].values.astype(np.float32)  # (num_samples,)

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        sample_id = self.ids[idx]

        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)

        # 添加序列长度维度 (seq_length=1)
        feature = feature.unsqueeze(0)  # 从 (num_features,) 变为 (1, num_features)

        return sample_id, feature, label


# 定义不含差分隐私的 LSTM 模型

class LSTMPredictor_noDP(nn.Module):
    def __init__(self):
        super(LSTMPredictor_noDP, self).__init__()
        # 定义 LSTM 层，输入为时间步的特征维度，隐藏层单元数，层数为可配置
        self.lstm = nn.LSTM(8, 64, 1, batch_first=True)
        # 定义全连接层，用于将 LSTM 的输出映射到最终的输出维度
        self.out = nn.Linear(64, 1)
        # 激活函数使用 ReLU
        self.act = nn.ReLU()

    def forward(self, x):
        # LSTM expects input of shape (batch_size, sequence_length, input_size)
        # LSTM 的输入形状为 (batch_size, 时间步, 特征维度)
        x, _ = self.lstm(x)
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        # 全连接层 + 激活函数
        x = self.act(x)
        x = self.out(x)
        return x

# 定义训练函数
def train_model(model, device, train_loader, optimizer, criterion, epoch, train_losses):
    model.train()
    running_loss = 0.0
    for batch_idx, (sample_ids, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device).unsqueeze(1)  # 确保标签形状为 [batch_size, 1]

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"Epoch [{epoch}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss/100)
    print(f"Epoch [{epoch}] 完成，平均损失: {avg_loss:.4f}")


# 定义测试函数
def test_model(model, device, test_loader, criterion, test_losses):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for sample_ids, data, target in test_loader:
            data, target = data.to(device), target.to(device).unsqueeze(1)  # 确保标签形状为 [batch_size, 1]
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    test_losses.append(avg_test_loss)
    print(f"测试集平均损失: {avg_test_loss:.4f}")
    return avg_test_loss


# 定义模型保存函数
def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")


# 定义模型加载函数
def load_model(path, device):
    model = LSTMPredictor_noDP()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    print(f"模型已加载并设置为评估模式。")
    return model


# 定义预测并保存结果的函数
def predict_and_save(model, device, dataset, save_path):
    """
    使用模型对数据集进行预测，并将预测结果保存到文件。

    :param model: 训练好的模型
    :param device: 设备（CPU 或 GPU）
    :param dataset: 要预测的数据集（ExcelDataset 类实例）
    :param save_path: 保存预测结果的文件路径（例如 'predictions.csv'）
    """
    model.eval()
    predictions = []
    sample_ids = []
    with torch.no_grad():
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for ids, data, _ in data_loader:
            data = data.to(device)
            outputs = model(data)
            outputs = outputs.cpu().numpy().flatten()
            predictions.extend(outputs)
            sample_ids.extend(ids.numpy())

    # 创建 DataFrame 保存预测结果
    df_predictions = pd.DataFrame({
        'ID': sample_ids,
        'Prediction': predictions
    })

    # 保存到 CSV 文件
    df_predictions.to_csv(save_path, index=False)
    print(f"预测结果已保存到 {save_path}")

def plot_loss(train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(8, 6))
    # plt.plot(epochs,train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, test_losses, label='Test Loss', marker='x')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()


# 主训练流程
# 主训练流程
def main():
    # 配置参数
    config = {
        "dataset_file_url": "/Users/cuitianyu/Desktop/1.xlsx",  # 请替换为您的数据文件路径
        "batch_size": 32,
        "epochs": 15,
        "lr": 0.001,
        "hidden_size": 64,
        "output_size": 1,
        "num_layers": 1,
        "test_split": 0.2,
        "random_seed": 42,
        "model_save_path": "LSTMPredictor.pth",
        "prediction_save_path": "predictions.csv",
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    }

    # 设置随机种子（可选）
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])

    # 定义数据转换
    feature_transform = transforms.Compose([
        ToTensor1D(),
    ])

    # 创建数据集实例
    dataset = ExcelDataset(file_path=config["dataset_file_url"], transform=feature_transform)

    # 划分训练集和测试集
    test_size = int(config["test_split"] * len(dataset))
    train_size = len(dataset) - test_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    print(f"训练集样本数: {len(train_dataset)}")
    print(f"测试集样本数: {len(test_dataset)}")

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    # 实例化模型
    model = LSTMPredictor_noDP()
    model.to(config["device"])

    # 定义损失函数和优化器
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    # 保存损失值
    train_losses = []
    test_losses = []

    # 训练和测试模型
    for epoch in range(1, config["epochs"] + 1):
        train_model(model, config["device"], train_loader, optimizer, criterion, epoch, train_losses)
        test_loss = test_model(model, config["device"], test_loader, criterion, test_losses)

    # 绘制损失曲线
    plot_loss(train_losses, test_losses)

    # 保存模型
    save_model(model, config["model_save_path"])

    # 加载模型并进行预测
    loaded_model = load_model(config["model_save_path"], config["device"])

    # 使用测试集进行预测并保存结果
    predict_and_save(loaded_model, config["device"], test_dataset, config["prediction_save_path"])


if __name__ == "__main__":
    main()
