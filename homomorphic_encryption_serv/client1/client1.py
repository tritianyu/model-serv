import models, torch, copy
import numpy as np
from server import Server
from dataset_read import *
import argparse, json
import socket
import pickle

# 客户端1的IP地址（根据实际情况修改）
client1_id = '127.0.0.1'
# 客户端2的IP地址（根据实际情况修改）
client2_id = '127.0.0.1'
# 聚合服务器的IP地址（根据实际情况修改）
server_id = '127.0.0.1'


class Client1(object):

	def __init__(self, conf, public_key, weights, data_x, data_y):

		self.conf = conf

		self.public_key = public_key

		self.local_model = models.LR_Model(public_key=self.public_key, w=weights, encrypted=True)

		self.data_x = data_x

		self.data_y = data_y

	def local_train(self, weights):

		print("----------客户端1开始训练----------")

		original_w = weights
		self.local_model.set_encrypt_weights(weights)

		neg_one = self.public_key.encrypt(-1)
		
		for e in range(self.conf["local_epochs"]):
			print("正在进行本地训练的轮数：", e+1)
			if e > 0 and e % 2 == 0:
				self.local_model.encrypt_weights = Server.re_encrypt(self.local_model.encrypt_weights)

			idx = np.arange(self.data_x.shape[0])
			batch_idx = np.random.choice(idx, self.conf['batch_size'], replace=False)
			#print(batch_idx)
			
			x = self.data_x[batch_idx]
			x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
			y = self.data_y[batch_idx].reshape((-1, 1))
			
			#print((0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one).shape)
			
			#print(x.transpose().shape)
			
			#assert(False)

			batch_encrypted_grad = x.transpose() * (0.25 * x.dot(self.local_model.encrypt_weights) + 0.5 * y.transpose() * neg_one)
			encrypted_grad = batch_encrypted_grad.sum(axis=1) / y.shape[0]
			
			for j in range(len(self.local_model.encrypt_weights)):
				self.local_model.encrypt_weights[j] -= self.conf["lr"] * encrypted_grad[j]
		print()
		weight_accumulators = []
		# print(models.decrypt_vector(Server.private_key, weights))
		for j in range(len(self.local_model.encrypt_weights)):
			weight_accumulators.append(self.local_model.encrypt_weights[j] - original_w[j])
		
		return weight_accumulators


class Client2(object):

	def __init__(self, conf, public_key, weights, data_x, data_y):

		self.conf = conf

		self.public_key = public_key

		self.local_model = models.LR_Model(public_key=self.public_key, w=weights, encrypted=True)

		self.data_x = data_x

		self.data_y = data_y

	def local_train(self, weights):

		print("----------客户端2开始训练----------")

		original_w = weights
		self.local_model.set_encrypt_weights(weights)

		neg_one = self.public_key.encrypt(-1)

		for e in range(self.conf["local_epochs"]):
			print("正在进行本地训练的轮数：", e + 1)
			if e > 0 and e % 2 == 0:
				self.local_model.encrypt_weights = Server.re_encrypt(self.local_model.encrypt_weights)

			idx = np.arange(self.data_x.shape[0])
			batch_idx = np.random.choice(idx, self.conf['batch_size'], replace=False)
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
				self.local_model.encrypt_weights[j] -= self.conf["lr"] * encrypted_grad[j]
		print()
		weight_accumulators = []
		# print(models.decrypt_vector(Server.private_key, weights))
		for j in range(len(self.local_model.encrypt_weights)):
			weight_accumulators.append(self.local_model.encrypt_weights[j] - original_w[j])

		return weight_accumulators


def client(ip):

	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	with open('./utils/conf.json', 'r') as f:
		conf = json.load(f)

	train_datasets, eval_datasets = read_dataset()
	train_size = train_datasets[0].shape[0]
	per_client_size = int(train_size / conf["no_models"])

	# 创建socket对象
	client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# 连接到服务器
	client_socket.connect((ip, 12345))

	# 接收服务器发来的初始化全局模型（使用特殊方法接收大数据）
	server_data = recv_data(client_socket)
	if not server_data:
		client_socket.close()
	initial_global_model = pickle.loads(server_data)
	client_1 = Client1(conf, Server.public_key, initial_global_model,
					   train_datasets[0][0 * per_client_size: (0 + 1) * per_client_size],
					   train_datasets[1][0 * per_client_size: (0 + 1) * per_client_size])


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


if __name__ == "__main__":
	import sys
	#client(sys.argv[1])  # 命令行参数传入服务器IP地址
	client(server_id)  # 直接指定服务器IP地址

