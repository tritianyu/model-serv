import requests

BASE_URL = "http://127.0.0.1:5001"  # 请根据实际情况修改


def send_data_and_get_response():
    # 构建要发送的JSON数据
    data = {
    "projectJobId": "1780926403611033645",
    "role": "server",
    "baseConfig": {
        "modelControlUrl": "127.0.0.1",
        "modelCalUrlList": [
            {
                "userId": "129",
                "url": "127.0.0.1",
                "isInitiator": False
            },
            {
                "userId": "666",
                "url": "127.0.0.1",
                "isInitiator": True
            },
            {
                "userId": "1",
                "url": "127.0.0.1",
                "isInitiator": False
            }
        ],
        "platGwUrl": "http://localhost:48080"
    },
    "modelParams": {
        "dataSet": {
            "projectOrgans": [
                {
                    "userId": "129",
                    "userType": 2,
                    "resource": [
                        {
                            "id": "1780926403611033602",
                            "dataSetName": "必填测试",
                            "dataSetDesc": "23",
                            "dataSetFilePath": "/Users/cuitianyu/Desktop/1.xlsx",
                            "dataSetFileName": None,
                            "dataSetFileUrl": "/Users/cuitianyu/Desktop/1.xlsx",
                            "visibility": 2,
                            "visibilityUsers": "admin",
                            "remark": None,
                            "status": None,
                            "creator": "129"
                        }
                    ]
                },
                {
                    "userId": "666",
                    "userType": 1,
                    "resource": [
                        {
                            "id": "1780926403611023602",
                            "dataSetName": "必填测试",
                            "dataSetDesc": "23",
                            "dataSetFilePath": "/Users/cuitianyu/Desktop/1.xlsx",
                            "dataSetFileName": None,
                            "dataSetFileUrl": "/Users/cuitianyu/Desktop/1.xlsx",
                            "visibility": 2,
                            "visibilityUsers": "admin",
                            "remark": None,
                            "status": None,
                            "creator": "129"
                        }
                    ]
                },
                {
                    "userId": "1",
                    "userType": 2,
                    "resource": [
                        {
                            "id": "1781142905056477185",
                            "dataSetName": "测试更新",
                            "dataSetDesc": "2324",
                            "dataSetFilePath": "/Users/cuitianyu/Desktop/1.xlsx",
                            "dataSetFileName": None,
                            "dataSetFileUrl": "/Users/cuitianyu/Desktop/1.xlsx",
                            "visibility": 1,
                            "visibilityUsers": "",
                            "remark": None,
                            "status": None,
                            "creator": "1"
                        }
                    ]
                }
            ]
        },
        "modelData": {
                "epochs": 15,
            "securityProtocol": "dp",
    "num_users": 1,
    "frac": 1,
    "bs": 64,
    "lr": 0.1,
    "lr_decay": 1,
    "momentum": 0.01,
    "model": "lstm", # cnn
    "dataset": "carbon",# mnist
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
    "user": [0, 1]
        }
    }
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
