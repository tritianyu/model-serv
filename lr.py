import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class LinearRegressionGD:
    """
    使用梯度下降法实现的线性回归模型。
    """

    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        """
        训练模型。

        参数:
        - X: 特征矩阵，形状为 (n_samples, n_features)
        - y: 目标变量，形状为 (n_samples,)
        """
        n_samples, n_features = X.shape
        # 初始化权重和偏置
        self.weights = np.zeros(n_features)
        self.bias = 0

        for epoch in range(self.epochs):
            # 预测值
            y_pred = np.dot(X, self.weights) + self.bias
            # 计算损失（均方误差）
            loss = np.mean(abs(y_pred - y))/100
            self.loss_history.append(loss)

            # 计算梯度
            dw = (2 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (2 / n_samples) * np.sum(y_pred - y)

            # 更新权重和偏置
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # 每100个epoch打印一次损失
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss:.4f}")

    def predict(self, X):
        """
        进行预测。

        参数:
        - X: 特征矩阵，形状为 (n_samples, n_features)

        返回:
        - y_pred: 预测值，形状为 (n_samples,)
        """
        return np.dot(X, self.weights) + self.bias


def normalize_data(input_file, output_file, feature_range=(-1, 1)):
    """
    读取Excel文件，去除第一列和最后一列，对特征进行归一化，并保存到新的Excel文件。

    参数:
    - input_file: 输入的Excel文件路径。
    - output_file: 输出的Excel文件路径。
    - feature_range: 归一化范围，默认为 (-1, 1)。

    返回:
    - X: 归一化后的特征矩阵。
    - y: 目标变量数组。
    """
    # 读取Excel文件
    df = pd.read_excel(input_file)
    # 去除第一列和最后一列
    features = df.iloc[:, 1:-1]
    target = df.iloc[:, -1]

    # 查看原始特征数据
    print("原始特征数据：")
    print(features.head())

    # 归一化处理
    scaler = MinMaxScaler(feature_range=feature_range)
    normalized_features = scaler.fit_transform(features)

    # 将归一化后的数据转换为DataFrame，并保留原来的列名
    normalized_df = pd.DataFrame(normalized_features, columns=features.columns)
    # 添加目标变量列
    normalized_df['target'] = target.values

    # 保存归一化后的数据到新的Excel文件
    normalized_df.to_excel(output_file, index=False)
    print(f"归一化后的数据已保存到 {output_file}")

    return normalized_features, target.values


def plot_loss(loss_history):
    """
    绘制训练过程中损失下降的曲线。

    参数:
    - loss_history: 损失历史列表。
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color='blue')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MAE)')
    plt.grid(True)
    plt.show()


def main():
    df = pd.read_excel('/Users/cuitianyu/Desktop/1.xlsx')
    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # 分割数据为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"训练集大小: {X_train.shape[0]} 样本")
    print(f"测试集大小: {X_test.shape[0]} 样本")

    # 初始化线性回归模型
    lr_model = LinearRegressionGD(learning_rate=0.01, epochs=15)

    # 训练模型
    lr_model.fit(X_train, y_train)

    # 预测
    y_pred = lr_model.predict(X_test)

    # 计算均方误差
    mse = np.mean((y_pred - y_test) ** 2)
    print(f"测试集均方误差 (MAE): {mse:.4f}")

    # 绘制训练损失下降曲线
    plot_loss(lr_model.loss_history)
    print(lr_model.loss_history)

    # 可选：绘制真实值与预测值的对比图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='green', alpha=0.6)
    plt.title('True Values vs Predictions')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 参考线
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
