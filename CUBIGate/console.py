import gradio as g
import requests
from datetime import datetime

url = "http://203.255.176.55:30004"
# API 접근을 위한 인증 과정
auth = {
    "client_id": "yqUF2zdLz1NlLc7p9",
    "client_secret": "WERMLRs6zZEFQFajr"
}
global headers
headers = {}

def authen():
    token = requests.post(url=f'{url}/api/v1/auth/client/token', json=auth).json()['access_token']
    global headers
    headers = {
        "Authorization": f"Bearer {token}"
    }
    return "Authenticated"


def train(epsilon: float, delta: float, iteration: int):
    global headers
    train_config = {
        "iterations": iteration,
        "epsilon": epsilon,
        "delta": delta
    }
    r = requests.post(url=f'{url}/api/v1/service_requests/dp_msv/train', json=train_config, headers=headers).json()

    # 이미 진행 중인 학습이 있을 경우 오류 발생
    if 'errors' in r.keys():
        return r['errors'][0]['message']
    else:
        global job_id
        job_id = r['job_id']
        print(f'train job ID: {job_id}')
        return r['message']

def check_status():
    global job_id
    global headers
    r = requests.get(url=f'{url}/api/v1/service_requests/dp_msv/status?job_id={job_id}', headers=headers).json()
    if r['status'] == "SUCCESS":
        global result
        result = eval(r['output'])['result']
        print(f'result: {result}')
        return "It's done.", datetime.now()
        
    elif r['status'] == 'EXECUTING':
        return "It's running.", datetime.now()
    elif r['status'] == "PENDING":
        return "It's waiting.", datetime.now()
    else:
        return f"""Something went wrong.
                    {r['details']}"""

def train_status():
    msg, time = check_status()
    if 'done' in msg:
        global base
        global result
        base = result['file_path']
    return msg, time

def disable(status):
    if status == "It's done." or 'wrong' in status:
        return g.Button("Check status", interactive=False, disable=True), g.Button("Generate", interactive=True)
    btn = g.Button("Check status")
    btn.click(
        fn=check_status, outputs=[status, g.Markdown()], every=10
    )
    return btn, g.Button("Generate", interactive=False)


def download():
    pass



def download_able(status):
    if 'done' in status:
        return g.Button("Download", )


def generate():
    global base
    global headers
    global job_id
    job_id = requests.post(url=f'{url}/api/v1/service_requests/dp_msv/generate?checkpoint_path={base}', headers=headers).json()
    while True:
        msg, time = check_status()
        if 'done' in msg:
            return msg
        elif 'wrong' in msg:
            return msg

with g.Blocks() as console:
    g.Button(value="Authenticate").click(fn=authen, outputs=[g.Markdown()])
    with g.Row():
        epsilon = g.Number(label="Epsilon", minimum=0)
        delta = g.Number(label='Delta', minimum=0)
        iteration = g.Slider(label='Iteration', minimum=2, maximum=25, value=2)
        output = g.Markdown()
    train_btn = g.Button("Train").click(
        fn=train, inputs=[epsilon, delta, iteration], outputs=[output] 
    )
    status = g.Markdown()
    train_status_btn = g.Button("Check status")
    train_status_btn.click(
        fn=train_status, outputs=[status, g.Markdown()], every=10
    )
    gen_btn = g.Button("Generate", interactive=False)
    gen_done = g.Markdown()
    gen_btn.click(
        fn=generate, outputs=[gen_done]
    )
    status.change(disable, inputs=status, outputs=[train_status_btn, gen_btn])
    gen_done.change
    

console.launch()