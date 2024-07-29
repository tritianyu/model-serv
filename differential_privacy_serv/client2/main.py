import random
import socket
import struct
import time
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import argparse, json
import pickle
import threading
import time

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid,cifar_noniid
from utils.options import args_parser
from models.Update import LocalUpdateDP, LocalUpdateDPSerial
from models.Nets import MLP, CNNMnist, CNNCifar, CNNFemnist, CharLSTM
from models.Fed import FedAvg, FedWeightAvg
from models.test import test_img
from utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule

server_ip = "127.0.0.1"
client2_ip = "127.0.0.1"


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
    datas = bytearray()
    bytes_received = 0
    while bytes_received < data_length:
        # 每次接收1024字节
        chunk = sock.recv(min(1024, data_length - bytes_received))
        if not chunk:
            return None
        datas.extend(chunk)
        bytes_received += len(chunk)

    return datas


if __name__ == '__main__':
    # parse args
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    lock = threading.Lock()

    args = args_parser()

    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # 创建socket对象
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到服务器
    client_socket.connect((server_ip, 12345))

    while True:
        # 接收数据
        data = recv_data(client_socket)
        if not data:
            break
        data = pickle.loads(data)

        model_and_loss = data[client2_ip]
        w = model_and_loss[1]
        local = model_and_loss[0]
        print("客户端 2 正在处理数据......")

        net_glob = CNNMnist(args=args).to(args.device)

        trained_model, loss_value = local.train(net=copy.deepcopy(w).to(args.device))
        data[client2_ip] = [model_and_loss[0], trained_model, loss_value]

        print("客户端 2 训练结束，准备上传模型......")
        # 发送处理后的数据
        send_data(client_socket, pickle.dumps(data))
        # client_socket.send(pickle.dumps(data))
        print("模型上传完毕\n")

    client_socket.close()
