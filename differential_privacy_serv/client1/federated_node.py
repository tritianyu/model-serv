import pickle
import random
import socket
import struct
import time
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import argparse, json
from flask import Flask, request, jsonify

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule
import requests
import threading


app = Flask(__name__)


@app.route('/')
def index():
    return 'Flask Web Service is running!'


@app.route('/process_data', methods=['POST'])
def process_data():
    # 获取请求中的JSON数据
    data = request.json
    role = data.get('role')
    if role == 'server':
        start_server(data)
    elif role == 'client':
        start_client(data)
    return jsonify({"message": "Role assigned and process started"}), 200


def start_server(config):
    app.logger.info(f"Server received config data: {config}")
    # 获取请求中的JSON数据
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    # model = CNNMnist(args)
    # model.load_state_dict(torch.load('DP_model.pth'))
    # model.eval()

    config["device"] = torch.device(
        'cuda:{}'.format(config["gpu"]) if torch.cuda.is_available() and config["gpu"] != -1 else 'cpu')
    dict_users = {}
    dataset_train, dataset_test = None, None

    # load dataset and split users
    if config["dataset"] == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        config["num_channels"] = 1
        # sample users
        if config["iid"]:
            dict_users = mnist_iid(dataset_train, config["num_users"])
        else:
            dict_users = mnist_noniid(dataset_train, config["num_users"])
    elif config["dataset"] == 'cifar':
        config["num_channels"] = 3
        trans_cifar_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        trans_cifar_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        dataset_train = datasets.CIFAR10('./data/cifar', train=True, download=True, transform=trans_cifar_train)
        dataset_test = datasets.CIFAR10('./data/cifar', train=False, download=True, transform=trans_cifar_test)
        if config["iid"]:
            dict_users = cifar_iid(dataset_train, config["num_users"])
        else:
            dict_users = cifar_noniid(dataset_train, config["num_users"])
    elif config["dataset"] == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        config["num_users"] = len(dict_users)
        if config["iid"]:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print(
                "Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    net_glob = None
    # build model
    if config["model"] == 'cnn' and config["dataset"] == 'cifar':
        net_glob = CNNCifar(args=config).to(config["device"])
    elif config["model"] == 'cnn' and (config["dataset"] == 'mnist'):
        net_glob = CNNMnist(args=config).to(config["device"])
    elif config["dataset"] == 'shakespeare' and config["model"] == 'lstm':
        net_glob = CharLSTM().to(config["device"])
    elif config["model"] == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=config["num_classes"]).to(config["device"])
    else:
        exit('Error: unrecognized model')

    # use opacus to wrap model to clip per sample gradient
    if config["dp_mechanism"] != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()  # 模型标识为训练状态

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(config["num_users"]))

    # training
    loss = 0
    acc_test = []
    acc_list = []
    loss_test = []
    loss_list = []
    if config["serial"]:
        clients = [LocalUpdateDPSerial(args=config, dataset=dataset_train, idxs=dict_users[i]) for i in
                   range(config["num_users"])]
    else:
        clients = [LocalUpdateDP(args=config, dataset=dataset_train, idxs=dict_users[i]) for i in
                   range(config["num_users"])]

    threads = []
    config['role'] = 'client'
    """for key in config:
        if "client" in key:
            client_url = f'http://{config["key"]}:5001/process_data'
            thread = threading.Thread(target=send_request, config=(client_url, data))
            threads.append(thread)
            thread.start()"""

    for key in config:
        if "client1" in key:
            client_url = f'http://{config[key]}:5002/process_data'
            thread = threading.Thread(target=send_request, args=(client_url, config))
            threads.append(thread)
            thread.start()
        if "client2" in key:
            client_url = f'http://{config[key]}:5003/process_data'
            thread = threading.Thread(target=send_request, args=(client_url, config))
            threads.append(thread)
            thread.start()
    # 创建socket对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定IP地址和端口号
    server_socket.bind((config["server_ip"], 12345))
    # 开始监听
    server_socket.listen(2)
    print("服务器启动，等待连接...")

    # 接受两个客户端的连接
    client1, addr1 = server_socket.accept()
    print(f'客户端{addr1}已连接')
    client2, addr2 = server_socket.accept()
    print(f'客户端{addr2}已连接')

    for iter in range(config["epochs"]):
        t_start = time.time()
        w_locals, loss_locals, weight_locals = [], [], []
        client_model_mapping = {
            config["client1_ip"]: [clients[0], net_glob, loss],
            config["client2_ip"]: [clients[1], net_glob, loss]
        }

        # 向客户端发送数据
        send_data(client1, pickle.dumps(client_model_mapping))
        send_data(client2, pickle.dumps(client_model_mapping))

        updated_data1 = pickle.loads(recv_data(client1))
        updated_data2 = pickle.loads(recv_data(client2))

        if config["client1_ip"] in updated_data1:
            client_model_mapping[config["client1_ip"]] = updated_data1[config["client1_ip"]]
            model_and_loss = client_model_mapping[config["client1_ip"]]
            w_locals.append(copy.deepcopy(model_and_loss[1]))
            loss_locals.append(copy.deepcopy(model_and_loss[2]))
            weight_locals.append(len(dict_users[0]))

        if config["client2_ip"] in updated_data2:
            client_model_mapping[config["client2_ip"]] = updated_data2[config["client2_ip"]]
            model_and_loss = client_model_mapping[config["client2_ip"]]
            w_locals.append(copy.deepcopy(model_and_loss[1]))
            loss_locals.append(copy.deepcopy(model_and_loss[2]))
            weight_locals.append(len(dict_users[1]))
        w_glob = FedWeightAvg(w_locals, weight_locals)
        net_glob.load_state_dict(w_glob)

        net_glob.eval()
        acc_t, loss_t = test_img(net_glob, dataset_test, config)
        t_end = time.time()
        print("Round {:3d},Testing accuracy: {:.2f},Loss：{:.5f}, Time:  {:.2f}s".format(iter, acc_t, loss_t,
                                                                                        t_end - t_start))

        acc_test.append(acc_t.item())
        acc_list.append(acc_t.item())
        loss_test.append(loss_t)
        loss_list.append(loss_t)

    # 关闭连接
    client1.close()
    client2.close()
    server_socket.close()

    model = CNNMnist(config)
    model.load_state_dict(torch.load('DP_model.pth', map_location=torch.device('cpu')))
    # 计算加密前后模型之间的Pearson相关性系数
    count = 0
    params_diff = []
    params_cnn = []
    params_cnn_1 = []
    for param, param_1 in zip(net_glob.parameters(), model.parameters()):
        params_cnn.append(param.data.cpu().view(-1).numpy())
        params_cnn_1.append(param_1.data.cpu().view(-1).numpy())

    for param, param_1 in zip(params_cnn, params_cnn_1):
        for p, p_1 in zip(param, param_1):
            diff = abs(p - p_1)
            if diff <= 0.00035:
                count = count + 1
            params_diff.append(diff)
    recognition_rate = (count / len(params_diff)) * 100
    print("可识别率：{:.3f}%".format(recognition_rate))
    # params_diff.sort()
    # print(params_diff[1092])

    filename = 'result/output_DP.json'
    data_dict = {'准确率%': acc_list, '损失': loss_list, '识别率%': recognition_rate}
    # 使用 'with' 语句打开文件，创建一个文件对象 file_obj
    with open('results/output_DP.json', 'w', encoding='utf-8') as file_obj:
        # 使用 json.dump() 将数据列表写入文件
        json.dump(data_dict, file_obj, indent=4, ensure_ascii=False)

    rootpath = './log'
    # if not os.path.exists(rootpath):
    #    os.makedirs(rootpath)
    accfile = open('results/DP_acc.dat', "w")

    for ac in acc_test:
        sac = str(ac)
        accfile.write(sac)
        accfile.write('\n')
    accfile.close()

    # plot accuracy curve
    # plt.figure()
    # plt.plot(range(len(acc_test)), acc_test)
    #
    # plt.ylabel('test accuracy')
    # plt.xlabel('rounds of training')
    # plt.savefig('results/DP_acc.png')

    # 绘制多曲线对比图
    # draw.draw_picture()

    # os.system('pause')
    # 保存模型
    torch.save(net_glob.state_dict(), 'global_model.pth')

    conf_file_path = 'results/output_DP.json'
    if os.path.exists(conf_file_path):
        with open(conf_file_path, 'r') as conf_file:
            result_data = json.load(conf_file)
    # 构建响应数据
    response = {
        "original_data": config,
        "processed_data": result_data,
        "message": "Data processed successfully"
    }
    # 返回JSON响应
    return jsonify(response), 200


def start_client(config):
    app.logger.info(f"Client received config data: {config}")
    # parse args
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    lock = threading.Lock()

    config["device"] = torch.device('cuda:{}'.format(config["gpu"]) if torch.cuda.is_available() and config["gpu"] != -1 else 'cpu')

    # 创建socket对象
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到服务器
    client_socket.connect((config["server_ip"], 12345))

    while True:
        # 接收数据
        data = recv_data(client_socket)
        if not data:
            break
        data = pickle.loads(data)

        model_and_loss = data[config["client2_ip"]]
        w = model_and_loss[1]
        local = model_and_loss[0]
        print("客户端 2 正在处理数据......")

        # net_glob = CNNMnist(args=config).to(config["device"])
        print(local)
        trained_model, loss_value = local.train(net=copy.deepcopy(w).to(config["device"]))
        data[config["client2_ip"]] = [model_and_loss[0], trained_model, loss_value]

        print("客户端 2 训练结束，准备上传模型......")
        # 发送处理后的数据
        send_data(client_socket, pickle.dumps(data))
        # client_socket.send(pickle.dumps(data))
        print("模型上传完毕\n")

    client_socket.close()


def get_local_ip():
    try:
        # 创建一个UDP连接
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # 连接到一个外部的IP地址，这里用的是Google的DNS服务器IP
        s.connect(("8.8.8.8", 80))
        # 获取本地IP地址
        local_ip = s.getsockname()[0]
        s.close()
    except Exception as e:
        local_ip = "Unable to get IP"
        print(f"Error: {e}")
    return local_ip


def send_request(client_url, data):
    try:
        response = requests.post(client_url, json=data)
        print(f"Response from {client_url}: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request to {client_url} failed: {e}")


def send_data(sock, data):
    # 计算数据长度并打包
    data_length = len(data)
    length_prefix = struct.pack('!I', data_length)

    # 发送数据长度
    sock.sendall(length_prefix)

    # 分块发送实际数据
    bytes_sent = 0
    while bytes_sent < data_length:
        # 每次发送1024字节
        chunk = data[bytes_sent:bytes_sent + 1024]
        sock.sendall(chunk)
        bytes_sent += len(chunk)


def recv_data(sock):
    # 接收数据长度
    length = sock.recv(4)
    if not length:
        return None
    data_length, = struct.unpack('!I', length)

    # 分块接收实际数据
    data = bytearray()
    bytes_received = 0
    while bytes_received < data_length:
        # 每次接收1024字节
        chunk = sock.recv(min(1024, data_length - bytes_received))
        if not chunk:
            return None
        data.extend(chunk)
        bytes_received += len(chunk)

    return data


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
