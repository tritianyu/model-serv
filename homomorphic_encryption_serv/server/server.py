import paillier
import matplotlib.pyplot as plt
import argparse, json
import models
import socket
import pickle
from dataset_read import *
from flask import Flask, request, jsonify

# 客户端1的IP地址（根据实际情况修改）
client1_id = '192.168.40.81'
# 客户端2的IP地址（根据实际情况修改）
client2_id = '192.168.40.82'
# 聚合服务器的IP地址（根据实际情况修改）
server_id = '192.168.40.80'


app = Flask(__name__)
# 首页路由，通常用来检查服务是否正常运行
@app.route('/')
def index():
    return 'Flask Web Service is running!'

class Server(object):
	
	public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)
	
	def __init__(self, conf, eval_dataset):
	
		self.conf = conf

		self.global_model = models.LR_Model(public_key=Server.public_key, w_size=self.conf["feature_num"]+1)

		self.eval_x = eval_dataset[0]
		
		self.eval_y = eval_dataset[1]
		
	def model_aggregate(self, weight_accumulator):
	
		for id, data in enumerate(self.global_model.encrypt_weights):
			
			update_per_layer = weight_accumulator[id] * self.conf["lambda"]
			
			self.global_model.encrypt_weights[id] = self.global_model.encrypt_weights[id] + update_per_layer
			
	
	def model_eval(self):
		
		
		total_loss = 0.0
		correct = 0
		dataset_size = 0
		
		batch_num = int(self.eval_x.shape[0]/self.conf["batch_size"])
		# print("本轮加密的全局模型参数：")
		# print(self.global_model.encrypt_weights[0].ciphertext())

		self.global_model.weights = models.decrypt_vector(Server.private_key, self.global_model.encrypt_weights)
		print("本轮全局模型参数：")
		print(self.global_model.weights)
	
		for batch_id in range(batch_num):
		
			x = self.eval_x[batch_id*self.conf["batch_size"]: (batch_id+1)*self.conf["batch_size"]]
			x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
			y = self.eval_y[batch_id*self.conf["batch_size"]: (batch_id+1)*self.conf["batch_size"]].reshape((-1, 1))

			dataset_size += x.shape[0]

			wxs = x.dot(self.global_model.weights)
			
			pred_y = [1.0 / (1 + np.exp(-wx)) for wx in wxs]
			
			#print(pred_y)
			
			pred_y = np.array([1 if pred > 0.5 else -1 for pred in pred_y]).reshape((-1, 1))
			
			#print(y)
			#print(pred_y)
			correct += np.sum(y == pred_y)

		#print(correct, dataset_size)
		acc = 100.0 * (float(correct) / float(dataset_size))
		# total_loss = total_loss / dataset_size

		return acc, self.global_model.weights
	
	@staticmethod
	def re_encrypt(w):
		return models.encrypt_vector(Server.public_key, models.decrypt_vector(Server.private_key, w))


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

# 处理接收JSON数据并返回处理结果的路由
@app.route('/process_data', methods=['POST'])
def process_data():
	parser = argparse.ArgumentParser(description='Federated Learning')
	parser.add_argument('-c', '--conf', dest='conf')
	args = parser.parse_args()
	with open('./utils/conf.json', 'r') as f:
		conf = json.load(f)
	# 获取请求中的JSON数据
	data = request.json
	# 根据接收到的数据修改配置文件内容
	for key, value in data.items():
		if key in conf:
			conf[key] = value
	# 将修改后的数据写回配置文件
	with open('./utils/conf.json', 'w') as conf_file:
		json.dump(conf, conf_file, indent=4)
	# 重新加载配置文件
	with open('./utils/conf.json', 'r') as f:
		conf = json.load(f)


	train_datasets, eval_datasets = read_dataset()

	server = Server(conf, eval_datasets)

	test_acc = []
	train_size = train_datasets[0].shape[0]
	per_client_size = int(train_size / conf["no_models"])


	# 创建socket对象
	server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	# 绑定IP地址和端口号
	server_socket.bind((server_id, 12345))
	# 开始监听
	server_socket.listen(2)
	print("服务器启动，等待连接...")

	# 接受两个客户端的连接
	client1, addr1 = server_socket.accept()
	print(f'客户端{addr1}已连接')
	client2, addr2 = server_socket.accept()
	print(f'客户端{addr2}已连接')

	# 为客户端创建好实例并发送给各个客户端
	client1_instance = Client1(conf, Server.public_key, server.global_model.encrypt_weights,
							  	train_datasets[0][0 * per_client_size: (0 + 1) * per_client_size],
								train_datasets[1][0 * per_client_size: (0 + 1) * per_client_size])
	client2_instance = Client2(conf, Server.public_key, server.global_model.encrypt_weights,
							  	train_datasets[0][1 * per_client_size: (1 + 1) * per_client_size],
								train_datasets[1][1 * per_client_size: (1 + 1) * per_client_size])

	instances = {client1_id: client1_instance, client2_id: client2_instance}

	# 发送大数据用特殊方法发送
	# client1.send(pickle.dumps(instances))
	# client2.send(pickle.dumps(instances))
	send_data(client1, pickle.dumps(instances))
	send_data(client2, pickle.dumps(instances))


	for e in range(conf["global_epochs"]):

		print(f"----------第{e+1}轮循环----------")

		server.global_model.encrypt_weights = models.encrypt_vector(Server.public_key,
																	models.decrypt_vector(Server.private_key,
																	server.global_model.encrypt_weights))
		# candidates = random.sample(clients, conf["k"])
		# candidates = selectDataset(clients, clients_id, conf["k"])
		weight_accumulator = [Server.public_key.encrypt(0.0)] * (conf["feature_num"] + 1)
		# print(server.global_model.weights)

		data = {client1_id: server.global_model.encrypt_weights, client2_id: server.global_model.encrypt_weights}

		# 向客户端发送全局模型(用特殊方法发送大数据)
		# client1.send(pickle.dumps(data))
		# client2.send(pickle.dumps(data))
		send_data(client1, pickle.dumps(data))
		send_data(client2, pickle.dumps(data))

		# 接收处理后的本地模型更新（用特殊方法接收大数据）
		# updated_data1 = pickle.loads(client1.recv(10240))
		# updated_data2 = pickle.loads(client2.recv(10240))
		updated_data1 = pickle.loads(recv_data(client1))
		updated_data2 = pickle.loads(recv_data(client2))

		# 更新全局模型
		if client1_id in updated_data1:
			diff = updated_data1[client1_id]
			for i in range(len(weight_accumulator)):
				weight_accumulator[i] = weight_accumulator[i] + diff[i]

		if client2_id in updated_data2:
			diff = updated_data2[client2_id]
			for i in range(len(weight_accumulator)):
				weight_accumulator[i] = weight_accumulator[i] + diff[i]

		server.model_aggregate(weight_accumulator)
		acc, global_model_weights = server.model_eval()

		test_acc.append(float(acc / 100))
		json_data = {"global_epochs": conf["global_epochs"], "Accuracy": test_acc,
					 "model_parameter": global_model_weights}

		if e == conf["global_epochs"] - 1:
			filename = './results/results_HE.json'
			with open(filename, 'w') as file_obj:
				json.dump(json_data, file_obj, indent=4)
			# 构建响应数据
			response = {
				"processed_data": json_data,
				"message": "Data processed successfully"
			}
			print("Global Epoch %d, acc: %f\n" % (e + 1, acc))
			# 返回JSON响应
			return jsonify(response), 200

		print("Global Epoch %d, acc: %f\n" % (e + 1, acc))

	# 关闭连接
	client1.close()
	client2.close()
	server_socket.close()

	plt.figure()
	x = range(1, len(test_acc) + 1, 1)
	plt.plot(x, test_acc, color='r', marker='.')
	plt.ylabel('Accuracy')
	plt.xlabel('Epochs')
	plt.title('federated learning with HE')
	plt.savefig('./results/acc.png')
	plt.show()


if __name__ == '__main__':
	app.run(debug=True)
