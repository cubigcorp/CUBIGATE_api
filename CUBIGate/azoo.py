import requests
import zipfile
import io
import sys
import time

url = "http://203.255.176.55:30004"

# API 접근을 위한 인증 과정
auth = {
    "client_id": "yqUF2zdLz1NlLc7p9",
    "client_secret": "WERMLRs6zZEFQFajr"
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
r = requests.post(url=f'{url}/api/v1/service_requests/dp_msv/train', json=train_config, headers=headers).json()

# 이미 진행 중인 학습이 있을 경우 오류 발생
if 'errors' in r.keys():
    print(r['errors'][0]['message'])
    sys.exit()

job_id = r['job_id']


print(f"Starting training with job ID: {job_id}")
# 상태 확인
base = ""
while True:
    r = requests.get(url=f'{url}/api/v1/service_requests/dp_msv/status?job_id={job_id}', headers=headers).json()
    if r['status'] == "SUCCESS":
        output = eval(r['output'])
        base = output['result']['file_path']
        break
    elif r['status'] == 'EXECUTING':
        print("It's running")
    elif r['status'] == "PENDING":
        print("It's waiting")
    else:
        print("Something went wrong.")
        sys.exit()
    time.sleep(10)

# 생성
job_id = requests.post(url=f'{url}/api/v1/service_requests/dp_msv/generate?checkpoint_path={base}', headers=headers).json()['job_id']
print(f"Generated images with job ID: {job_id}")

time.sleep(5)

# 저장
r = requests.get(url=f'{url}/api/v1/service_requests/dp_msv/generate/download?job_id={job_id}', headers=headers)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./result/cookie/gen")