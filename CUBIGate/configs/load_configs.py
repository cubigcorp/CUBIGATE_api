import os
import json


# Load configs from json file base on ENV variable
def load_configs():
    env = os.environ.get('ENV', 'dev')
    with open(f'configs/config.{env}.json',  encoding='utf-8') as f:
        configs = json.load(f)
    return configs