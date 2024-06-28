import requests

BASE_URL = "http://127.0.0.1:5002"  # 请根据实际情况修改


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
                                "dataSetFilePath": None,
                                "dataSetFileName": None,
                                "dataSetFileUrl": "/Users/cuitianyu/model-serv/homomorphic_encryption_serv/client2/data/breast.csv",
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
                                "dataSetFilePath": None,
                                "dataSetFileName": None,
                                "dataSetFileUrl": "/Users/cuitianyu/model-serv/homomorphic_encryption_serv/server/data/breast.csv",
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
                                "dataSetFilePath": None,
                                "dataSetFileName": None,
                                "dataSetFileUrl": "/Users/cuitianyu/model-serv/homomorphic_encryption_serv/client1/data/breast.csv",
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
                "modelId": "LR",
                "securityProtocol": "同态加密",
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
                "feature_num": 30
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
