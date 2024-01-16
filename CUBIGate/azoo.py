import requests
import zipfile
import io

url = "http://203.255.176.55:30004"

# API 접근을 위한 인증 과정
auth = {
    "client_id": "qo9XKhCGYdCRTf0HT",
    "client_secret": "MEj1BmqsqaYNuunRz"
}
token = requests.post(url=f'{url}/api/v1/auth/client/token', json=auth).json()['access_token']
headers = {
    "Authorization": f"Bearer {token}"
}

# 학습
train_config = {
    "iterations": 2,
    "epsilon": 1,
    "delta": 0
}
base = requests.post(url=f'{url}/api/v1/services/dp_msv/train', json=train_config, headers=headers).json()['checkpoint_path']

# 생성
gen_config = {
    "checkpoint_path": base
}
r = requests.get(url=f'{url}/api/v1/services/dp_msv/generate', json=gen_config, headers=headers)
print(r.json())
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./result/cookie/gen")