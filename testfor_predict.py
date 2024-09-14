import requests

BASE_URL = "http://127.0.0.1:5001"


def send_data_and_get_response():
    # 构建要发送的JSON数据
    data = {'modelParams': {'modelType': 'he', 'dataSet': [{'userId': '130', 'userType': '1', 'userName': 'admin01', 'providerOrganIds': [{'id': '1796001202615996417', 'dataSetName': '1测试流成', 'dataSetDesc': '12321', 'dataSetFilePath': '/Users/cuitianyu/Desktop/2.xlsx', 'dataSetFileName': None, 'dataSetFileUrl': None, 'visibility': 1, 'visibilityUsers': '', 'remark': None, 'status': None, 'createTime': 1717035029000, 'creator': '130'}]}]}, 'projectJobId': '1833040448765624322', 'modelServName': '推理测试2'}
    # 向 /process_data 端点发送POST请求，附带JSON数据
    response = requests.post(f"{BASE_URL}/predict", json=data)

    # 打印响应文本内容以便调试
    print("Response text:", response.text)

    # 打印响应状态码和JSON格式的响应内容
    try:
        print("POST /process_data:", response.status_code, response.json())
    except requests.exceptions.JSONDecodeError:
        print("Failed to parse JSON response")


if __name__ == "__main__":
    send_data_and_get_response()
