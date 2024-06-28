import numpy as np
import struct


def read_dataset(file_path):
	data_X, data_Y = [], []
	
	with open(file_path) as fin:
		for line in fin:
			data = line.split(',')
			data_X.append([float(e) for e in data[:-1]])
			if int(data[-1]) == 1:
				data_Y.append(1)
			else:
				data_Y.append(-1)
	
	data_X = np.array(data_X)
	data_Y = np.array(data_Y)
	

	
	idx = np.arange(data_X.shape[0])
	np.random.shuffle(idx)
	
	train_size = int(data_X.shape[0]*0.8)
	
	train_x = data_X[idx[:train_size]]
	train_y = data_Y[idx[:train_size]]
	
	eval_x = data_X[idx[train_size:]]
	eval_y = data_Y[idx[train_size:]]
	
	return (train_x, train_y), (eval_x, eval_y)


def selectDataset(clients, clients_id, idx_list):
	candidates = []
	for client in clients:
		for idx in idx_list:
			if clients_id[str(client)] == idx:
				candidates.append(client)
				break
	return candidates


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
	lengthbuf = sock.recv(4)
	if not lengthbuf:
		return None
	data_length, = struct.unpack('!I', lengthbuf)

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
