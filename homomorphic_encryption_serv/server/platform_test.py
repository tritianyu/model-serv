import requests

BASE_URL = "http://127.0.0.1:5002"  # 请根据实际情况修改


def send_data_and_get_response():
    # 构建要发送的JSON数据
    data = {'projectJobId': '1833040448765624322', 'role': 'server', 'baseConfig': {'modelControlUrl': '127.0.0.1', 'modelCalUrlList': [{'url': '127.0.0.1', 'userId': '131', 'isInitiator': False}, {'url': '127.0.0.1', 'userId': '132', 'isInitiator': False}, {'url': '127.0.0.1', 'userId': '130', 'isInitiator': True}], 'platGwUrl': 'http://localhost:48080'}, 'modelParams': {'dataSet': {'projectOrgans': [{'userId': '131', 'userType': '2', 'resource': [{'id': '1796004654112866306', 'dataSetName': '用户2数据', 'dataSetDesc': '3423423', 'dataSetFilePath': '/workspace/3.xlsx', 'dataSetFileName': None, 'dataSetFileUrl': None, 'visibility': 2, 'visibilityUsers': '', 'remark': None, 'status': None, 'creator': '131'}]}, {'userId': '132', 'userType': '2', 'resource': [{'id': '1796004918421127169', 'dataSetName': '用户3数据', 'dataSetDesc': '23423', 'dataSetFilePath': '/workspace/3.xlsx', 'dataSetFileName': None, 'dataSetFileUrl': None, 'visibility': 3, 'visibilityUsers': '', 'remark': None, 'status': None, 'creator': '132'}]}, {'userId': '130', 'userType': '1', 'resource': [{'id': '1796001202615996417', 'dataSetName': '1测试流成', 'dataSetDesc': '12321', 'dataSetFilePath': '/workspace/3.xlsx', 'dataSetFileName': None, 'dataSetFileUrl': None, 'visibility': 1, 'visibilityUsers': '', 'remark': None, 'status': None, 'creator': '130'}]}]}, 'modelData': {'modelId': 'LR', 'securityProtocol': 'he', 'no_models': '3', 'global_epochs': '20', 'k': '[1,2]', 'local_epochs': '2', 'batch_size': '32', 'lr': '0.001', 'momentum': '0.0001', 'lambda': '0.2', 'feature_num': '8', 'epochs': '100', 'dpEpsilon': '20', 'numUsers': '64', 'user': '[0,1]', 'gaussian': None}}}

    # 向 /process_data 端点发送POST请求，附带JSON数据
    response = requests.post(f"{BASE_URL}/process_data", json=data)

    # 打印响应文本内容以便调试
    print("Response text:", response.text)

    # 打印响应状态码和JSON格式的响应内容
    try:
        print("POST /process_data:", response.status_code, response.json())
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse JSON response")


if __name__ == "__main__":
    send_data_and_get_response()
