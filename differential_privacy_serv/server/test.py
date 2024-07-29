import requests

BASE_URL = "http://127.0.0.1:5001"  # 请根据实际情况修改


def send_data_and_get_response():
    # 构建要发送的JSON数据
    data = {
    "epochs": 15,
    "num_users": 2,
    "frac": 1,
    "bs": 64,
    "lr": 0.1,
    "lr_decay": 1,
    "momentum": 0.01,
    "model": "cnn",
    "dataset": "mnist",
    "iid": True,
    "num_classes": 10,
    "num_channels": 1,
    "gpu": 0,
    "dp_mechanism": "Gaussian",
    "dp_epsilon": 20,
    "dp_delta": 1e-5,
    "dp_clip": 10,
    "dp_sample": 1,
    "serial": False,
    "serial_bs": 128,
    "batch_size": 64,
    "user": [0, 1],
    "role": "server",
    "client1_ip": "127.0.0.1",
    "client2_ip": "127.0.0.1",
    "server_ip": "127.0.0.1"
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
