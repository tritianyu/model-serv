import json, requests

url = "http://127.0.0.1:5002/diagnose"
# payload = {
#     "projectJobId": "1833040448765624322",
#     "modelType": "lr",
#     "currentHparams": {
#         "lr": 0.001,
#         "lambda": 0.1,
#         "local_epochs": 5,
#         "global_epochs": 20
#     }
# }
# payload = {
#     "projectJobId": "1780926403611033645",
#     "modelType": "dnn",
#     "currentHparams": {
#         "lr": 0.001,
#         "batch_size": 32,
#         "epochs": 15,
#         "dp_epsilon": 20
#     }
# }
payload = {'projectJobId': '1965978600148619266', 'modelId': '1965978751336501249', 'modelType': 'lr', 'currentHparams': {'global_epochs': 10.0, 'local_epochs': 2.0, 'batch_size': 32.0, 'lr': 0.0005, 'frac': 1.0, 'bs': 64.0, 'lr_decay': 0.0, 'momentum': 0.0001, 'num_classes': 10.0, 'num_channels': 1.0, 'lambda': 0.2, 'feature_num': 8.0, 'epochs': 15.0, 'dp_epsilon': 20.0, 'dp_delta': 1e-05, 'dp_clip': 10.0, 'dp_sample': 1.0, 'serial_bs': 128.0}}
resp = requests.post(url, json=payload, timeout=30)
print(resp.status_code)
print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
