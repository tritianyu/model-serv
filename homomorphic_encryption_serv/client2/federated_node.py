import argparse, json, socket, pickle
from flask import Flask, request, jsonify
import models, torch, copy
import numpy as np
import paillier
from dataset_read import *
import matplotlib.pyplot as plt
import requests
import threading

app = Flask(__name__)


class Server(object):
    public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)

    def __init__(self, conf, eval_dataset):

        self.conf = conf

        self.global_model = models.LR_Model(public_key=Server.public_key, w_size=self.conf["modelParams"]["modelData"]["feature_num"] + 1)

        self.eval_x = eval_dataset[0]

        self.eval_y = eval_dataset[1]

    def model_aggregate(self, weight_accumulator):

        for id, data in enumerate(self.global_model.encrypt_weights):
            update_per_layer = weight_accumulator[id] * self.conf["modelParams"]["modelData"]["lambda"]

            self.global_model.encrypt_weights[id] = self.global_model.encrypt_weights[id] + update_per_layer

    def model_eval(self):

        total_loss = 0.0
        correct = 0
        dataset_size = 0

        batch_num = int(self.eval_x.shape[0] / self.conf["modelParams"]["modelData"]["batch_size"])
        # print("本轮加密的全局模型参数：")
        # print(self.global_model.encrypt_weights[0].ciphertext())

        self.global_model.weights = models.decrypt_vector(Server.private_key, self.global_model.encrypt_weights)
        print("本轮全局模型参数：")
        print(self.global_model.weights)

        for batch_id in range(batch_num):
            x = self.eval_x[batch_id * self.conf["modelParams"]["modelData"]["batch_size"]: (batch_id + 1) * self.conf["modelParams"]["modelData"]["batch_size"]]
            x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
            y = self.eval_y[batch_id * self.conf["modelParams"]["modelData"]["batch_size"]: (batch_id + 1) * self.conf["modelParams"]["modelData"]["batch_size"]].reshape(
                (-1, 1))

            dataset_size += x.shape[0]

            wxs = x.dot(self.global_model.weights)

            pred_y = [1.0 / (1 + np.exp(-wx)) for wx in wxs]

            # print(pred_y)

            pred_y = np.array([1 if pred > 0.5 else -1 for pred in pred_y]).reshape((-1, 1))

            # print(y)
            # print(pred_y)
            correct += np.sum(y == pred_y)

        # print(correct, dataset_size)
        acc = 100.0 * (float(correct) / float(dataset_size))
        # total_loss = total_loss / dataset_size

        return acc, self.global_model.weights

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

            # print((0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one).shape)

            # print(x.transpose().shape)

            # assert(False)

            batch_encrypted_grad = x.transpose() * (
                    0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one)
            encrypted_grad = batch_encrypted_grad.sum(axis=1) / y.shape[0]

            for j in range(len(self.local_model.encrypt_weights)):
                self.local_model.encrypt_weights[j] -= self.conf["modelParams"]["modelData"]["lr"] * encrypted_grad[j]

        weight_accumulators = []
        # print(models.decrypt_vector(Server.private_key, weights))
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

    role = data.get('role')
    if role == 'server':
        response = start_server(data)
        return jsonify(response), 200
    elif role == 'client':
        start_client(data)
        return jsonify({"message": "Client process started"}), 200
    else:
        return jsonify({"message": "Invalid role"}), 400


def start_server(config):

    app.logger.info(f"Server received config data: {config}")

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((config["baseConfig"]["modelControlUrl"], 12345))
    server_socket.listen(len(config["baseConfig"]["modelCalUrlList"]))
    print("服务器启动，等待连接...")

    config['role'] = 'client'
    threads = []

    for entry in config['baseConfig']['modelCalUrlList']:
        client_url = ""

        user_id = int(entry['userId'])

        url = entry['url']

        is_initiator = entry['isInitiator']
        if is_initiator:
            dataset_file_url = None
            for organ in config["modelParams"]["dataSet"]["projectOrgans"]:
                if int(organ["userId"]) == user_id:
                    dataset_file_url = organ["resource"][0]["dataSetFileUrl"]
                    break
            if dataset_file_url:
                train_datasets, eval_datasets = read_dataset(dataset_file_url)
                server = Server(config, eval_datasets)
                test_acc = []
                train_size = len(train_datasets)
                per_client_size = int(train_size / config["modelParams"]["modelData"]["no_models"])
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
            if user_id == 1:
                client_url = f'http://{url}:5001/process_data'
            elif user_id == 129:
                client_url = f'http://{url}:5003/process_data'
            thread = threading.Thread(target=send_request, args=(client_url, config))
            threads.append(thread)
            thread.start()

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
            "data_slice": f"train_datasets[0][{i * per_client_size}:{(i + 1) * per_client_size}]",
        "data_slice_y": f"train_datasets[1][{i * per_client_size}:{(i + 1) * per_client_size}]",
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

    for e in range(config["modelParams"]["modelData"]["global_epochs"]):
        print(f"Global Epoch {e + 1}")
        # 重新加密全局模型权重
        server.global_model.encrypt_weights = models.encrypt_vector(Server.public_key,
                                                                    models.decrypt_vector(Server.private_key,
                                                                                          server.global_model.encrypt_weights))
        weight_accumulator = [Server.public_key.encrypt(0.0)] * (config["modelParams"]["modelData"]["feature_num"] + 1)
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
                     "model_parameter": global_model_weights}

        if e == config["modelParams"]["modelData"]["global_epochs"] - 1:
            filename = './results/results_HE.json'
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


def start_client(config):

    # 获取本地ip
    # client1_id = get_local_ip()
    client1_id = '127.0.0.1'
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
            dataset_file_url = organ['resource'][0]['dataSetFileUrl']
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5003)
