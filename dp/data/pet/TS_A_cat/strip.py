import os

path = os.path.dirname(os.path.abspath(__file__))
for img in os.listdir(path):
    if not img.endswith('.jpg'):
        continue
    org = os.path.join(path, img)
    split = img.split('_')
    striped = '_'.join(split[2:])
    str_img = os.path.join(path, striped)
    os.replace(org, str_img)