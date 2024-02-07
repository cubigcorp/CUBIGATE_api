import gradio as g
import requests

url = "http://223.130.131.19:30004"
# API 접근을 위한 인증 과정
auth = {
    "client_id": "yqUF2zdLz1NlLc7p9",
    "client_secret": "WERMLRs6zZEFQFajr"
}
global headers
headers = {}

result_path = 'result'

def authen():
    token = requests.post(url=f'{url}/api/v1/auth/client/token', json=auth).json()['access_token']
    global headers
    headers = {
        "Authorization": f"Bearer {token}"
    }
    return "Authenticated"


def train(epsilon: float, delta: float, iterations: int):
    global headers
    train_config = {
        "iterations": iterations,
        "epsilon": epsilon,
        "delta": delta
    }
    r = requests.post(url=f'{url}/api/v1/service_requests/dp_msv/train', json=train_config, headers=headers).json()

    # 이미 진행 중인 학습이 있을 경우 오류 발생
    if 'errors' in r.keys():
        return r['errors'][0]['message']
    elif r['message'] == 'success':
        return 'Training Started'
    else:
        return str(r)

def check_status(service: str):
    global headers
    r = requests.get(url=f'{url}/api/v1/service_requests/dp_msv/{service}/status', headers=headers).json()
    if r['job_status']['status'] == "SUCCESS":
        return "done."
        
    elif r['job_status']['status'] == 'EXECUTING':
        return "running."
    elif r['job_status']['status'] == "PENDING":
        return "waiting."
    elif r['job_status']['status'] == "FAILED":
        return "failed."
    else:
        return f"""Something went wrong.
                    {r['details']}"""

def train_status():
    msg = check_status('train')
    return f'Train {msg}'


def gen_status():
    msg = check_status('generate')
    return f'Generation {msg}'




def download():
    global headers
    r = requests.get(url=f'{url}/api/v1/service_requests/dp_msv/generate/download', headers=headers, stream=True)
    with open(f'{result_path}/generated.zip', 'wb') as f:
        for chunk in r.iter_content(chunk_size=128):
            f.write(chunk)
    return f'{result_path}/generated.zip'


def generate():
    global headers
    r = requests.post(url=f'{url}/api/v1/service_requests/dp_msv/generate', headers=headers).json()
    # 이미 진행 중인 학습이 있을 경우 오류 발생
    if 'errors' in r.keys():
        return r['errors'][0]['message']
    elif r['message'] == 'success':
        return 'Generation Started.'

with g.Blocks() as console:
    g.Button(value="Authenticate").click(fn=authen, outputs=[g.Markdown()])
    with g.Row(equal_height=True):
        with g.Column():
            epsilon = g.Number(label="Epsilon", minimum=0, value=1.0)
            delta = g.Number(label='Delta', minimum=0, value=0.01)
            iteration = g.Slider(label='Iteration', minimum=2, maximum=20, value=2)
            train_out = g.Markdown()
            train_btn = g.Button("Train")
            train_btn.click(
                fn=train, inputs=[epsilon, delta, iteration], outputs=[train_out] 
            )
            train_status_btn = g.Button("Check Train Status")
            train_status_btn.click(
                fn=train_status, outputs=[g.Markdown()]
            )
        with g.Column():
            gen_btn = g.Button("Generate")
            gen_done = g.Markdown()
            gen_btn.click(
                fn=generate, outputs=[gen_done]
            )
            gen_status_btn = g.Button("Check Generate Status")
            gen_status_btn.click(
                fn=gen_status, outputs=[g.Markdown()]
            )
            down_btn = g.Button("Download")
            down_btn.click(
                fn=download, outputs=g.File()
            )
    

console.launch()