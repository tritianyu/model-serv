import requests

BASE_URL = "http://127.0.0.1:5002"  # 请根据实际情况修改


def send_data_and_get_response():
    # 构建要发送的JSON数据
    data = {
    "client1_id": "127.0.0.1",
    "client2_id": "127.0.0.1",
    "server_id": "127.0.0.1",
    "port": 12345,
    "no_models": 3,
    "global_epochs": 3,
    "local_epochs": 2,
    "k": [
        2,
        3
    ],
    "batch_size": 32,
    "lr": 0.15,
    "momentum": 0.0001,
    "lambda": 0.5,
    "feature_num": 30,
    "role": "server"
}
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
