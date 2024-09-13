import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 设定文件路径
input_file = '/Users/cuitianyu/Desktop/1.xlsx'
output_file = 'normalized.xlsx'

# 读取Excel文件
df = pd.read_excel(input_file)
features = df.iloc[:, 1:-1]

# 查看特征数据
print("原始特征数据：")
print(features.head())

# 归一化处理
scaler = MinMaxScaler(feature_range=(-1, 1))
normalized_features = scaler.fit_transform(features)

# 将归一化后的数据转换为DataFrame，并保留原来的列名
normalized_df = pd.DataFrame(normalized_features, columns=features.columns)

# 保存归一化后的数据到新的Excel文件
normalized_df.to_excel(output_file, index=False)

print(f"归一化后的数据已保存到 {output_file}")