import matplotlib.pyplot as plt
import matplotlib as mpl


def openfile(filepath):
    file = open(filepath)
    y = []
    while 1:
        line = file.readline()
        if line.rstrip('\n') == '':
            break
        y.append(float(line.rstrip('\n')))
        if not line:
            break
        pass
    file.close()
    return y


def openfile(filename):
    # 这里假设数据文件是简单的文本文件，每行一个浮点数
    with open(filename, 'r') as file:
        data = [float(line.strip()) for line in file]
    return data


def draw_picture():
    mpl.use('TkAgg')
    plt.figure()
    plt.ylabel('Testing Accuracy')
    plt.xlabel('Global Round')

    # 读取数据
    y_no_DP = openfile('./results/no_DP_acc.dat')
    y_DP = openfile('./results/DP_acc.dat')

    # 找到较短的数据长度以设置x轴范围
    min_length = min(len(y_no_DP), len(y_DP))

    # 根据数据长度调整x轴范围，并画图
    plt.plot(range(min_length), y_no_DP[:min_length], label=r'No DP')
    plt.plot(range(min_length), y_DP[:min_length], label=r'DP')

    plt.title('Gaussian')
    plt.legend()
    plt.savefig('./results/gaussian.png')
    plt.show()
