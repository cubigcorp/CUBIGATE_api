import requests
import zipfile
import io
import sys
import time

url = "http://223.130.131.19:30004"

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
    "epsilon": 1.0,
    "delta": 1.e-2
}
r = requests.post(url=f'{url}/api/v1/service_requests/dp_msv/train', json=train_config, headers=headers).json()

# 이미 진행 중인 학습이 있을 경우 오류 발생
if 'errors' in r.keys():
    print(r['errors'][0]['message'])
    sys.exit()

train_job_id = r['job_id']


print(f"Starting training with job ID: {train_job_id}")
# 상태 확인
def check_status(service):
    while True:
        r = requests.get(url=f'{url}/api/v1/service_requests/dp_msv/{service}/status', headers=headers).json()
        if r['job_status']['status'] == "SUCCESS":
            print("It's done.")
            output = eval(r['job_status']['output'])
            return output['result']
            
        elif r['job_status']['status'] == 'EXECUTING':
            print("It's running.")
        elif r['job_status']['status'] == "PENDING":
            print("It's waiting.")
        elif r['job_status']['status'] == 'FAILED':
            print('It failed')
        else:
            print("Something went wrong.")
            print(r['details'])
            sys.exit()
        time.sleep(10)


base = check_status('train')['file_path']


# 생성
gen_job_id = requests.post(url=f'{url}/api/v1/service_requests/dp_msv/generate', headers=headers).json()['job_id']
print(f"Generating images with job ID: {gen_job_id}")

result = check_status('generate')

# 저장
r = requests.get(url=f'{url}/api/v1/service_requests/dp_msv/generate/download', headers=headers)
z = zipfile.ZipFile(io.BytesIO(r.content))
z.extractall("./result/cookie/gen")