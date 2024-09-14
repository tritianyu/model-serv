from homomorphic_encryption_serv.server import models, paillier
from homomorphic_encryption_serv.server.dataset_read import *
import matplotlib.pyplot as plt
import pickle
import random
import socket
import time

import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import os
import json
from flask import Flask, request, jsonify

from differential_privacy_serv.server.utils.sampling import mnist_iid, mnist_noniid, cifar_iid, cifar_noniid
from differential_privacy_serv.server.models.Update import LocalUpdateDP, LocalUpdateDPSerial, ExcelDataset
from differential_privacy_serv.server.models.Nets import CNNMnist, CNNCifar, CharLSTM, LSTMPredictor, LSTMPredictor_noDP
from differential_privacy_serv.server.models.Fed import FedAvg, FedWeightAvg
from differential_privacy_serv.server.models.test import test_img
from differential_privacy_serv.server.utils.dataset import FEMNIST, ShakeSpeare
from opacus.grad_sample import GradSampleModule
import requests
import threading


app = Flask(__name__)
client1_id = '127.0.0.1'


class Server(object):
    public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

    def __init__(self, conf, eval_dataset):

        self.conf = conf

        self.global_model = models.LR_Model(public_key=Server.public_key, w_size=int(self.conf["modelParams"]["modelData"]["feature_num"]) + 1)

        self.eval_x = eval_dataset[0]

        self.eval_y = eval_dataset[1]

    def model_aggregate(self, weight_accumulator):

        for id, data in enumerate(self.global_model.encrypt_weights):
            update_per_layer = weight_accumulator[id] * self.conf["modelParams"]["modelData"]["lambda"]

            self.global_model.encrypt_weights[id] = self.global_model.encrypt_weights[id] + update_per_layer

    def model_eval(self):
        from sklearn.metrics import mean_absolute_error
        total_relative_error = 0.0
        total_loss = 0.0
        correct = 0
        dataset_size = 0

        batch_num = int(self.eval_x.shape[0] / self.conf["modelParams"]["modelData"]["batch_size"])
        # print("本轮加密的全局模型参数：")
        # print(self.global_model.encrypt_weights[0].ciphertext())

        self.global_model.weights = models.decrypt_vector(Server.private_key, self.global_model.encrypt_weights)
        print("本轮全局模型参数：")
        print(self.global_model.weights)
        all_pred = []
        all_true = []
        for batch_id in range(batch_num):
            x = self.eval_x[batch_id * self.conf["modelParams"]["modelData"]["batch_size"]: (batch_id + 1) * self.conf["modelParams"]["modelData"]["batch_size"]]
            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            y = self.eval_y[batch_id * self.conf["modelParams"]["modelData"]["batch_size"]: (batch_id + 1) * self.conf["modelParams"]["modelData"]["batch_size"]].reshape(
                (-1, 1))

            dataset_size += x.shape[0]

            wxs = x.dot(self.global_model.weights)

            # pred_y = [1.0 / (1 + np.exp(-wx)) for wx in wxs]
            #
            # # print(pred_y)
            #
            # pred_y = np.array([1 if pred > 0.5 else -1 for pred in pred_y]).reshape((-1, 1))
            pred_y = wxs
            # print(y)
            # print(pred_y)
            # correct += np.sum(y == pred_y)
            # 计算相对误差
            pred_y = wxs.flatten()
            true_y = y.flatten()
            all_pred.extend(pred_y)
            all_true.extend(true_y)
            # print(y)
            # print(pred_y)
            # correct += np.sum(y == pred_y)
            # 计算相对误差
            relative_error = np.abs((pred_y - y) / y)
            total_relative_error += np.sum(relative_error)

        # print(correct, dataset_size)
        # acc = 100.0 * (float(correct) / float(dataset_size))
        # total_loss = total_loss / dataset_size
        # 计算平均相对误差
        mae = mean_absolute_error(all_true, all_pred)

        return mae, self.global_model.weights

    @staticmethod
    def re_encrypt(w):
        return models.encrypt_vector(Server.public_key, models.decrypt_vector(Server.private_key, w))


class Client(object):

    def __init__(self, conf, public_key, weights, data_x, data_y, num):

        self.conf = conf

        self.num = num

        self.public_key = public_key

        self.local_model = models.LR_Model(public_key=self.public_key, w=weights, encrypted=True)

        self.data_x = data_x

        self.data_y = data_y

    def local_train(self, weights):

        print(f"----------客户端{self.num}开始训练----------")

        original_w = weights
        self.local_model.set_encrypt_weights(weights)

        neg_one = self.public_key.encrypt(-1)

        for e in range(self.conf["modelParams"]["modelData"]["local_epochs"]):
            print("正在进行本地训练的轮数：", e + 1)
            if e > 0 and e % 2 == 0:
                self.local_model.encrypt_weights = Server.re_encrypt(self.local_model.encrypt_weights)

            idx = np.arange(self.data_x.shape[0])
            if len(idx) == 0:
                raise ValueError(f"Client {self.num} has no data to train on.")

            batch_idx = np.random.choice(idx, self.conf["modelParams"]["modelData"]['batch_size'], replace=False)
            # print(batch_idx)

            x = self.data_x[batch_idx]
            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            y = self.data_y[batch_idx].reshape((-1, 1))

            batch_encrypted_grad = x.transpose() * (
                    0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one)
            encrypted_grad = batch_encrypted_grad.sum(axis=1) / y.shape[0]

            for j in range(len(self.local_model.encrypt_weights)):
                self.local_model.encrypt_weights[j] -= self.conf["modelParams"]["modelData"]["lr"] * encrypted_grad[j]

        weight_accumulators = []
        for j in range(len(self.local_model.encrypt_weights)):
            weight_accumulators.append(self.local_model.encrypt_weights[j] - original_w[j])

        return weight_accumulators


# 首页路由，通常用来检查服务是否正常运行
@app.route('/')
def index():
    return 'Flask Web Service is running!'


@app.route('/process_data', methods=['POST'])
def process_data():

    # 获取请求中的JSON数据
    data = request.json
    app.logger.info(f"Received request data: {data}")
    encryption = data['modelParams']['modelData'].get('securityProtocol')
    if encryption == 'he':
        role = data.get('role')
        if role == 'server':
            response = start_he_server(data)
            return jsonify(response), 200
        elif role == 'client':
            start_he_client(data)
            return jsonify({"message": "Client process started"}), 200
        else:
            return jsonify({"message": "Invalid role"}), 400
    elif encryption == 'dp':
        role = data.get('role')
        if role == 'server':
            response = start_dp_server(data)
            return jsonify(response), 200
        elif role == 'client':
            start_dp_client(data)
            return jsonify({"message": "Client process started"}), 200
        else:
            return jsonify({"message": "Invalid role"}), 400
    else:
        return jsonify({"message": "Invalid security protocol"}), 400


def start_he_server(config):

    app.logger.info(f"Server received config data: {config}")
    config["modelParams"]["modelData"]["no_models"] = int(config["modelParams"]["modelData"]["no_models"])
    config["modelParams"]["modelData"]["global_epochs"] = int(config["modelParams"]["modelData"]["global_epochs"])
    config["modelParams"]["modelData"]["local_epochs"] = int(config["modelParams"]["modelData"]["local_epochs"])
    config["modelParams"]["modelData"]["batch_size"] = int(config["modelParams"]["modelData"]["batch_size"])
    config["modelParams"]["modelData"]["lr"] = float(config["modelParams"]["modelData"]["lr"])
    config["modelParams"]["modelData"]["feature_num"] = int(config["modelParams"]["modelData"]["feature_num"])
    config["modelParams"]["modelData"]["lambda"] = float(config["modelParams"]["modelData"]["lambda"])

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # server_socket.bind((config["baseConfig"]["modelControlUrl"], 12345))
    try:
        server_socket.bind(("0.0.0.0", 12345))
        server_socket.listen(len(config["baseConfig"]["modelCalUrlList"]))
        print("服务器启动，等待连接...")

        config['role'] = 'client'
        threads = []

        for entry in config['baseConfig']['modelCalUrlList']:
            client_url = ""

            user_id = int(entry['userId'])

            url = str(entry['url'])

            if entry["url"] == config["baseConfig"]["modelControlUrl"]:
                print("这是服务端")
                dataset_file_url = None
                for organ in config["modelParams"]["dataSet"]["projectOrgans"]:
                    if int(organ["userId"]) == user_id:
                        dataset_file_url = organ["resource"][0]["dataSetFilePath"]
                        break
                if dataset_file_url:
                    train_datasets, eval_datasets = read_dataset(dataset_file_url)
                    server = Server(config, eval_datasets)
                    test_acc = []
                    # train_size = train_datasets[0].shape[0]
                    # per_client_size = int(train_size / config["modelParams"]["modelData"]["no_models"])
                else:
                    app.logger.error(f"No dataset_file_url found for user_id: {user_id}")
                    continue
            else:
                # 实际用这个
                """
                    client_url = f'http://{url}:5000/process_data'
                thread = threading.Thread(target=send_request, args=(client_url, config))
                threads.append(thread)
                thread.start()"""
                print("这是客户端")
                if user_id == 131:
                    client_url = f'http://{url}:5001/process_data'
                    print(client_url)
                elif user_id == 132:
                    client_url = f'http://{url}:5003/process_data'
                    print(client_url)
                thread = threading.Thread(target=send_request, args=(client_url, config))
                threads.append(thread)
                thread.start()
                app.logger.info(config)
    except Exception as e:
        app.logger.error(f"Error occurred while starting he server: {e}")
        server_socket.close()
        print("服务器和所有客户端连接已关闭")

    clients = []
    instances = {}
    connected_clients = 0
    # 过滤出非isInitiator的model
    non_initiators = [model_url for model_url in config["baseConfig"]["modelCalUrlList"] if
                      not model_url["isInitiator"]]
    # 动态接受客户端连接并创建实例
    for i, model_url in enumerate(non_initiators):
        client_socket, addr = server_socket.accept()
        connected_clients += 1
        print(f'客户端{addr}已连接')

        client_instance_config = {
            "config": config,
            "public_key": Server.public_key,
            "data_slice": f"train_datasets[0][{i} * per_client_size:{(i + 1)} * per_client_size]",
        "data_slice_y": f"train_datasets[1][{i} * per_client_size:{(i + 1)}* per_client_size]",
            "num": i + 1
        }
        clients.append(client_socket)
        instances[model_url["url"]] = client_instance_config
    app.logger.info(f"Total connected clients: {connected_clients}")

    # 创建包含初始全局模型和配置数据的字典
    initial_data = {
        "initial_global_model": server.global_model.encrypt_weights,
        "config_data": instances
    }
    # 发送初始化数据到客户端
    app.logger.info("Sending initial data to clients")
    for client_socket in clients:
        send_data(client_socket, pickle.dumps(initial_data))
        app.logger.info(f"Sent data to client: {client_socket}")

    for e in range(int(config["modelParams"]["modelData"]["global_epochs"])):
        print(f"Global Epoch {e + 1}")
        # 重新加密全局模型权重
        server.global_model.encrypt_weights = models.encrypt_vector(Server.public_key,
                                                                    models.decrypt_vector(Server.private_key,
                                                                                          server.global_model.encrypt_weights))
        weight_accumulator = [Server.public_key.encrypt(0.0)] * (int(config["modelParams"]["modelData"]["feature_num"]) + 1)
        data_to_send = {model_url["url"]: server.global_model.encrypt_weights for model_url in config["baseConfig"]["modelCalUrlList"]}
        for client_socket in clients:
            send_data(client_socket, pickle.dumps(data_to_send))
            app.logger.info(f"Sent global model to client: {client_socket}")

        for client_socket in clients:
            updated_data = pickle.loads(recv_data(client_socket))
            for model_url in config["baseConfig"]["modelCalUrlList"]:
                if model_url["url"] in updated_data:
                    diff = updated_data[model_url["url"]]
                    for i in range(len(weight_accumulator)):
                        weight_accumulator[i] += diff[i]

        server.model_aggregate(weight_accumulator)
        acc, global_model_weights = server.model_eval()

        test_acc.append(float(acc / 100))
        json_data = {"global_epochs": config["modelParams"]["modelData"]["global_epochs"], "Accuracy": test_acc,
                     "model_parameter": global_model_weights, "recognize_rate": 0.0}

        if e == int(config["modelParams"]["modelData"]["global_epochs"]) - 1:
            filename = f'saved_models/he_models/{config["projectJobId"]}results_HE.json'
            with open(filename, 'w') as file_obj:
                json.dump(json_data, file_obj, indent=4)
            response = {"processed_data": json_data, "message": "Data processed successfully"}
            print(f"Global Epoch {e + 1}, acc: {acc}\n")
            return response

        print(f"Global Epoch {e + 1}, acc: {acc}\n")

    for client_socket in clients:
        client_socket.close()
    server_socket.close()

    plt.figure()
    x = range(1, len(test_acc) + 1, 1)
    plt.plot(x, test_acc, color='r', marker='.')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.title('Federated Learning with HE')
    plt.savefig('./results/acc.png')
    plt.show()


def start_he_client(config):

    # 获取本地ip
    # client1_id = get_local_ip()

    matching_entry = None
    for entry in config['baseConfig']['modelCalUrlList']:
        if entry['url'] == client1_id:
            matching_entry = entry
            break
    if not matching_entry:
        app.logger.error(f"Not find matching client url")
    user_id = matching_entry['userId']
    for organ in config['modelParams']['dataSet']['projectOrgans']:
        if organ['userId'] == user_id:
            dataset_file_url = organ['resource'][0]['dataSetFilePath']
            break

    train_datasets, eval_datasets = read_dataset(dataset_file_url)

    train_size = train_datasets[0].shape[0]
    per_client_size = int(train_size / config["modelParams"]["modelData"]["no_models"])

    # 创建socket对象
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到服务器
    client_socket.connect((config["baseConfig"]["modelControlUrl"], 12345))
    print(f"{client1_id}已连接服务器")
    # 接收服务器发来的客户端实例（使用特殊方法接收大数据）
    # client_instance = client_socket.recv(10240)
    initial_data = recv_data(client_socket)
    print("已接受数据")
    if not initial_data:
        client_socket.close()
    initial_data = pickle.loads(initial_data)
    initial_global_model = initial_data["initial_global_model"]
    config_data = initial_data["config_data"]
    # 根据接受到的配置实例化本地模型
    client_config = config_data[client1_id]
    data_slice = client_config["data_slice"]
    data_slice_y = client_config["data_slice_y"]
    print(data_slice)
    # 使用 eval() 函数来获取实际的数据分片
    train_data_x = eval(data_slice)
    print(train_data_x)
    train_data_y = eval(data_slice_y)
    client_1 = Client(
        client_config["config"],
        client_config["public_key"],
        initial_global_model,
        train_data_x,
        train_data_y,
        client_config["num"]
    )

    print("已加载实例")

    # client_1 = Client(conf, Server.public_key, data['192.168.40.81'],train_datasets[0][0*per_client_size: (0+1)*per_client_size],train_datasets[1][0*per_client_size: (0+1)*per_client_size])
    while True:
        # 接收服务器发来的全局模型参数，准备训练（使用特殊方法接收大数据）
        # data = client_socket.recv(10240)
        data = recv_data(client_socket)
        if not data:
            break
        data = pickle.loads(data)
        print("服务器广播的数据：", data)

        global_model = data[client1_id]
        local_model = client_1.local_train(global_model)
        data[client1_id] = local_model
        # 发送本地模型更新（使用特殊方法发送大数据）
        # client_socket.send(pickle.dumps(local_model))
        send_data(client_socket, pickle.dumps(data))
    client_socket.close()


def start_dp_server(config):

    app.logger.info(f"Server received config data: {config}")
    # 获取请求中的JSON数据
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)
    # config替换为modelParams

    # model = CNNMnist(args)
    # model.load_state_dict(torch.load('DP_model.pth'))
    # model.eval()
    matching_entry = None
    for entry in config['baseConfig']['modelCalUrlList']:
        if entry['url'] == client1_id:
            matching_entry = entry
            break
    if not matching_entry:
        app.logger.error(f"Not find matching client url")
    user_id = matching_entry['userId']
    for organ in config['modelParams']['dataSet']['projectOrgans']:
        if organ['userId'] == user_id:
            dataset_file_url = organ['resource'][0]['dataSetFilePath']
            break

    config["device"] = torch.device(
        'cuda:{}'.format(config["modelParams"]["modelData"]["gpu"]) if torch.cuda.is_available() and config["modelParams"]["modelData"]["gpu"] != -1 else 'cpu')
    # load dataset and split users
    if config["modelParams"]["modelData"]["dataset"] == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        config["modelParams"]["modelData"]["num_channels"] = 1
        # sample users
        if config["modelParams"]["modelData"]["iid"]:
            dict_users = mnist_iid(dataset_train, config["modelParams"]["modelData"]["num_users"])
        else:
            dict_users = mnist_noniid(dataset_train, config["modelParams"]["modelData"]["num_users"])
    elif config["modelParams"]["modelData"]["dataset"] == 'cifar':
        config["modelParams"]["modelData"]["num_channels"] = 3
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
        if config["modelParams"]["modelData"]["iid"]:
            dict_users = cifar_iid(dataset_train, config["modelParams"]["modelData"]["num_users"])
        else:
            dict_users = cifar_noniid(dataset_train, config["modelParams"]["modelData"]["num_users"])
    elif config["modelParams"]["modelData"]["dataset"] == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        config["modelParams"]["modelData"]["num_users"] = len(dict_users)
        if config["modelParams"]["modelData"]["iid"]:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print(
                "Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif config["modelParams"]["modelData"]["dataset"] == 'carbon':
        class ToTensor1D(object):
            def __call__(self, feature):
                return torch.tensor(feature, dtype=torch.float32)
        feature_transform = transforms.Compose([
            ToTensor1D(),
        ])
        dataset_train = ExcelDataset(file_path=dataset_file_url, transform=feature_transform)
        dataset_test = ExcelDataset(file_path=dataset_file_url, transform=feature_transform)
    else:
        exit('Error: unrecognized dataset')
    # img_size = dataset_train[0][0].shape

    net_glob = None
    # build model
    if config["modelParams"]["modelData"]["model"] == 'cnn' and config["modelParams"]["modelData"]["dataset"] == 'cifar':
        net_glob = CNNCifar(args=config["modelParams"]["modelData"]).to(config["device"])
    elif config["modelParams"]["modelData"]["model"] == 'cnn' and (config["modelParams"]["modelData"]["dataset"] == 'mnist'):
        net_glob = CNNMnist(args=config["modelParams"]["modelData"]).to(config["device"])
    elif config["modelParams"]["modelData"]["dataset"] == 'shakespeare' and config["modelParams"]["modelData"]["model"] == 'lstm':
        net_glob = CharLSTM().to(config["device"])
    elif config["modelParams"]["modelData"]["model"] == 'lstm':
        net_glob = LSTMPredictor()
    # elif config["modelParams"]["modelData"]["model"] == 'mlp':
    #     len_in = 1
    #     for x in img_size:
    #         len_in *= x
    #     net_glob = MLP(dim_in=len_in, dim_hidden=64, dim_out=config["modelParams"]["modelData"]["num_classes"]).to(config["device"])
    else:
        exit('Error: unrecognized model')

    # use opacus to wrap model to clip per sample gradient
    if config["modelParams"]["modelData"]["dp_mechanism"] != 'no_dp':
        net_glob = GradSampleModule(net_glob)
    print(net_glob)
    net_glob.train()  # 模型标识为训练状态

    # copy weights
    w_glob = net_glob.state_dict()
    all_clients = list(range(config["modelParams"]["modelData"]["num_users"]))

    # training
    loss = 0
    acc_test = []
    acc_list = []
    loss_test = []
    loss_list = []
    # if config["modelParams"]["modelData"]["serial"]:
    #     clients = [LocalUpdateDPSerial(args=config["modelParams"]["modelData"], dataset=dataset_train, idxs=dict_users[i]) for i in
    #                range(config["modelParams"]["modelData"]["num_users"])]
    # else:
    #     clients = [LocalUpdateDP(args=config["modelParams"]["modelData"], dataset=dataset_train, idxs=dict_users[i]) for i in
    #                range(config["modelParams"]["modelData"]["num_users"])]

    # 创建socket对象
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 绑定IP地址和端口号
    server_socket.bind(("0.0.0.0", 12345))
    # 开始监听
    server_socket.listen(len(config["baseConfig"]["modelCalUrlList"]))
    print("服务器启动，等待连接...")

    threads = []
    config['role'] = 'client'
    config.pop('device', None)
    for entry in config['baseConfig']['modelCalUrlList']:
        client_url = ""

        user_id = int(entry['userId'])

        url = entry['url']

        is_initiator = entry['isInitiator']
        if is_initiator:
            continue
        else:
            # 实际用这个
            """
            client_url = f'http://{url}:5000/process_data'
            thread = threading.Thread(target=send_request, args=(client_url, config))
            threads.append(thread)
            thread.start()"""
            if user_id == 1:
                client_url = f'http://{url}:5002/process_data'
            elif user_id == 129:
                client_url = f'http://{url}:5003/process_data'
            thread = threading.Thread(target=send_request, args=(client_url, config))
            print(config)
            threads.append(thread)
            thread.start()

    client_socket_list = []
    connected_clients = 0
    # 过滤出非isInitiator的model
    non_initiators = [model_url for model_url in config["baseConfig"]["modelCalUrlList"] if
                      not model_url["isInitiator"]]
    # 动态接受客户端连接
    for i, model_url in enumerate(non_initiators):
        client_socket, addr = server_socket.accept()
        connected_clients += 1
        print(f'客户端{addr}已连接')
        client_socket_list.append(client_socket)
    app.logger.info(f"Total connected clients: {connected_clients}")

    for iter in range(config["modelParams"]["modelData"]["epochs"]):
        t_start = time.time()
        w_locals, loss_locals, weight_locals = [], [], [0, 0]
        client_model_mapping = {}
        for i, model_url in enumerate(non_initiators):
            client_model_mapping[model_url["url"]] = [weight_locals[i], net_glob, loss]

        # 向客户端发送数据
        for client_socket in client_socket_list:
            send_data(client_socket, pickle.dumps(client_model_mapping))
            app.logger.info(f"Sent data to client: {client_socket_list}")
        # send_data(client1, pickle.dumps(client_model_mapping))
        # send_data(client2, pickle.dumps(client_model_mapping))
        for client_socket in client_socket_list:
            updated_data = pickle.loads(recv_data(client_socket))
            for i, model_url in enumerate(non_initiators):
                if model_url["url"] in updated_data:
                    client_model_mapping[model_url["url"]] = updated_data[model_url["url"]]
                    model_and_loss = client_model_mapping[model_url["url"]]
                    # 只在 w_locals 中添加对应的元素，确保长度与 weight_locals 一致
                    if len(weight_locals) == len(w_locals):  # 检查长度是否相等
                        w_locals[len(weight_locals) - 1] = copy.deepcopy(model_and_loss[1])  # 替换最后一个元素
                    else:
                        w_locals.append(copy.deepcopy(model_and_loss[1]))  # 添加一个元素
                    loss_locals.append(copy.deepcopy(model_and_loss[2]))
                    weight_locals[i] = model_and_loss[0]

        # updated_data1 = pickle.loads(recv_data(client1))
        # updated_data2 = pickle.loads(recv_data(client2))
        w_glob = FedWeightAvg(w_locals, weight_locals)
        net_glob.load_state_dict(w_glob)

        net_glob.eval()
        # acc_t, loss_t = test_img(net_glob, dataset_test, config["modelParams"]["modelData"])
        loss_t = test_img(net_glob, dataset_test, config["modelParams"]["modelData"])
        t_end = time.time()
        # print("Round {:3d},Testing accuracy: {:.2f},Loss：{:.5f}, Time:  {:.2f}s".format(iter, acc_t, loss_t,
        #                                                                                 t_end - t_start))
        print("Round {:3d},Loss：{:.5f}, Time:  {:.2f}s".format(iter, loss_t, t_end - t_start))
        # acc_test.append(acc_t.item())
        # acc_list.append(acc_t.item())
        loss_test.append(loss_t)
        loss_list.append(loss_t)

    # 关闭连接
    for client_socket in client_socket_list:
        client_socket.close()

    server_socket.close()

    model = LSTMPredictor_noDP()
    model.load_state_dict(torch.load('LSTMPredictor.pth', map_location=torch.device('cpu')))
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
    torch.save(net_glob.state_dict(), f'saved_models/dp_models/{config["projectJobId"]}global_model.pth')

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
    return response


def start_dp_client(config):

    # get local ip
    local_client_ip = '127.0.0.1'
    app.logger.info(f"Client received config data: {config}")
    # parse args
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.cuda.manual_seed(123)

    lock = threading.Lock()

    config["device"] = torch.device('cuda:{}'.format(config["modelParams"]["modelData"]["gpu"]) if torch.cuda.is_available() and config["modelParams"]["modelData"]["gpu"] != -1 else 'cpu')
    # load dataset and split users
    matching_entry = None
    for entry in config['baseConfig']['modelCalUrlList']:
        if entry['url'] == client1_id:
            matching_entry = entry
            break
    if not matching_entry:
        app.logger.error(f"Not find matching client url")
    user_id = matching_entry['userId']
    for organ in config['modelParams']['dataSet']['projectOrgans']:
        if organ['userId'] == user_id:
            dataset_file_url = organ['resource'][0]['dataSetFilePath']
            break
    if config["modelParams"]["modelData"]["dataset"] == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('./data/mnist/', train=False, download=True, transform=trans_mnist)
        config["modelParams"]["modelData"]["num_channels"] = 1
        # sample users
        if config["modelParams"]["modelData"]["iid"]:
            dict_users = mnist_iid(dataset_train, config["modelParams"]["modelData"]["num_users"])
        else:
            dict_users = mnist_noniid(dataset_train, config["modelParams"]["modelData"]["num_users"])
    elif config["modelParams"]["modelData"]["dataset"] == 'cifar':
        config["modelParams"]["modelData"]["num_channels"] = 3
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
        if config["modelParams"]["modelData"]["iid"]:
            dict_users = cifar_iid(dataset_train, config["modelParams"]["modelData"]["num_users"])
        else:
            dict_users = cifar_noniid(dataset_train, config["modelParams"]["modelData"]["num_users"])
    elif config["modelParams"]["modelData"]["dataset"] == 'shakespeare':
        dataset_train = ShakeSpeare(train=True)
        dataset_test = ShakeSpeare(train=False)
        dict_users = dataset_train.get_client_dic()
        config["modelParams"]["modelData"]["num_users"] = len(dict_users)
        if config["modelParams"]["modelData"]["iid"]:
            exit('Error: ShakeSpeare dataset is naturally non-iid')
        else:
            print(
                "Warning: The ShakeSpeare dataset is naturally non-iid, you do not need to specify iid or non-iid")
    elif config["modelParams"]["modelData"]["dataset"] == 'carbon':
        class ToTensor1D(object):
            def __call__(self, feature):
                return torch.tensor(feature, dtype=torch.float32)
        feature_transform = transforms.Compose([
            ToTensor1D(),
        ])
        dataset_train = ExcelDataset(file_path=dataset_file_url, transform=feature_transform)
        dataset_test = ExcelDataset(file_path=dataset_file_url, transform=feature_transform)
        print(dataset_train)
        num_users = config["modelParams"]["modelData"]["num_users"]
        if config["modelParams"]["modelData"]["iid"]:
            dict_users = dataset_iid(dataset_train, num_users)
        else:
            dict_users = dataset_noniid(dataset_train, num_users)

    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # 创建socket对象
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 连接到服务器
    client_socket.connect((config["baseConfig"]["modelControlUrl"], 12345))

    while True:
        # 接收数据
        data = recv_data(client_socket)
        if not data:
            break
        data = pickle.loads(data)

        model_and_loss = data[local_client_ip]
        w = model_and_loss[1]

        if config["modelParams"]["modelData"]["serial"]:
            local = LocalUpdateDPSerial(args=config["modelParams"]["modelData"], dataset=dataset_train, idxs=dict_users[0])
        else:
            local = LocalUpdateDP(args=config["modelParams"]["modelData"], dataset=dataset_train, idxs=dict_users[0])
        print("客户端 2 正在处理数据......")

        # net_glob = CNNMnist(args=config).to(config["device"])
        trained_model, loss_value = local.train(net=copy.deepcopy(w).to(config["device"]))
        data[local_client_ip] = [len(dict_users[0]), trained_model, loss_value]

        print("客户端 2 训练结束，准备上传模型......")
        # 发送处理后的数据
        send_data(client_socket, pickle.dumps(data))
        # client_socket.send(pickle.dumps(data))
        print("模型上传完毕\n")

    client_socket.close()



def dataset_iid(dataset, num_users):
    """
    Sample I.I.D. client data from the given dataset
    :param dataset: PyTorch Dataset object
    :param num_users: number of users
    :return: dict of image index lists for each user
    """
    num_items = int(len(dataset) / num_users)
    all_indices = np.arange(len(dataset))
    np.random.shuffle(all_indices)
    dict_users = {}
    for i in range(num_users):
        start_idx = i * num_items
        end_idx = start_idx + num_items
        dict_users[i] = set(all_indices[start_idx:end_idx])
    return dict_users
def dataset_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from the given dataset
    :param dataset: PyTorch Dataset object
    :param num_users: number of users
    :return: dict of data index lists for each user
    """
    # Sort data indices by labels (assuming dataset returns (data, label))
    targets = [dataset[i][1] for i in range(len(dataset))]
    indices = np.arange(len(dataset))
    indices_targets = np.vstack((indices, targets))
    indices_targets = indices_targets[:, indices_targets[1, :].argsort()]
    indices = indices_targets[0, :]

    # Divide sorted indices into shards (e.g., 2 shards per user)
    num_shards = num_users * 2
    shards_size = int(len(dataset) / num_shards)
    shards = [indices[i * shards_size:(i + 1) * shards_size] for i in range(num_shards)]

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    shard_indices = np.random.permutation(num_shards)

    # Assign shards to users
    for i in range(num_users):
        shard_ids = shard_indices[i * 2:(i + 1) * 2]
        user_indices = np.concatenate([shards[shard_id] for shard_id in shard_ids])
        dict_users[i] = set(user_indices)
    return dict_users

def send_request(client_url, data):
    try:
        response = requests.post(client_url, json=data)
        print(f"Response from {client_url}: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request to {client_url} failed: {e}")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)


