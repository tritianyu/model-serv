
# -*- coding: utf-8 -*-
# Python version: 3.9


import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):
    """
    从MNIST数据集中独立同分布地采样客户端数据
    :param dataset: MNIST数据集对象
    :param num_users: 期望的用户数量
    :return: 包含图像索引的字典
    """
    dict_users = {}  # 存储用户索引和对应图像索引的字典
    num_items = int(len(dataset) / num_users)  # 每个用户应拥有的图像数量
    all_idxs = [i for i in range(len(dataset))]  # 所有图像的索引列表

    for i in range(num_users):
        # 从所有索引中随机选择num_items个索引，无放回
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))

        # 从all_idxs中移除已分配给用户的索引，确保不重复选择
        all_idxs = list(set(all_idxs) - dict_users[i])

    return dict_users


def mnist_noniid(dataset, num_users):
    """
    从MNIST数据集中采样非独立同分布的客户端数据
    :param dataset: MNIST数据集对象
    :param num_users: 期望的用户数量
    :return: 包含图像索引的字典
    """
    dict_users = {}  # 存储用户索引和对应图像索引的字典
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))  # 划分为shards的数量和每个shard的图像数量
    idx_shard = [i for i in range(num_shards)]  # shards的索引列表
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 初始化用户字典，每个用户对应一个空的图像索引数组
    idxs = np.arange(num_shards * num_imgs)  # 所有图像的索引数组
    labels = dataset.targets.numpy()  # 所有图像的标签数组

    # 将图像按标签进行排序
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # 划分和分配图像索引
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))  # 从shards中随机选择2个shard
        idx_shard = list(set(idx_shard) - rand_set)  # 从可选的shards中移除已经选中的shard
        for rand in rand_set:
            # 将选中的shard的图像索引添加到用户的图像索引数组中
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)

    return dict_users


def fashion_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def fashion_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    labels = dataset.targets.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users

def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    dict_users = {}
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = num_users * 2, int(len(dataset) / (num_users * 2))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    idxs = np.arange(num_shards * num_imgs)
    # labels = dataset.train_labels.numpy()
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    return dict_users


#  if __name__ == '__main__':
    #trans_fashion_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
   # dataset_train = datasets.FashionMNIST('../data/fashion-mnist', train=True, download=True,
                                         # transform=trans_fashion_mnist)
    # num = 100
    # d = mnist_iid(dataset_train, num)
    # path = '../data/fashion_iid_100clients.dat'
    # file = open(path, 'w')
    # for idx in range(num):
    #     for i in d[idx]:
    #         file.write(str(i))
    #         file.write(',')
    #     file.write('\n')
    # file.close()
    # trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    # dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
    # print(fashion_iid(dataset_train, 1000)[0])


