import os
import argparse
from typing import Dict, List
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import numpy as np
import json

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Path of the base data directory."
    )
    parser.add_argument(
        '--pretrained_model',
        type=str,
        required=True,
        help="Name of the pretrained model to use."
    )
    parser.add_argument(
        '--device',
        type=int,
        required=True,
        help="Index of the GPU device to put everything on."
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        required=True,
        help="Number of classes."
    )
    parser.add_argument(
        '--result_dir',
        type=str,
        required=False,
        default='Path of the directory where the result will be saved.'
    )
    parser.add_argument(
        '--dp',
        required=False,
        action='store_true',
        help="Whether the dataset is dp or non-dp"
    )
    args = parser.parse_args()
    return args

def make_images(dir: str) -> List:
    data = []
    for (root, dirs, files) in os.walk(dir):
        for file in files:
            if file.split('.')[-1] not in ['jpg', 'png']:
                continue
            full = os.path.join(root, file)
            data.append(full)     
    return data

def get_texts(config: Dict) -> List:
    texts = []
    base = config['base']
    for label in config['labels']:
        text = base.replace('LABEL', label)
        texts.append(text)
    return texts

def predict(images: List, texts: List, device_num: int) -> Dict:
    checkpoint = "openai/clip-vit-large-patch14"
    device = f'cuda:{device_num}'
    processor = CLIPProcessor.from_pretrained(checkpoint)
    model = CLIPModel.from_pretrained(checkpoint)
    model.to(device)
    predictions = {}
    for image_path in images:
        image = Image.open(image_path)
        inputs = processor(
            text = texts,
            padding = True,
            images=image,
            return_tensors='pt'
        ).to(device)
        outputs = model(**inputs)
        logits = outputs.logits_per_image.detach()
        probs = logits.softmax(dim=1)[0]
        label = torch.argmax(probs).item()
        predictions[image_path] = {
            'logits' : logits.detach().cpu().tolist(),
            'probs' : probs.detach().cpu().tolist(),
            'label': int(label)
        }
    return predictions

def get_label(pred: Dict, labels: List):
    pred_labels = []
    true_labels = []

    for img in pred.keys():
        temp = img.split('/')
        if len(temp) < 2:
            continue
        true_labels.append(img.split('/')[-2])
        pred_labels.append(labels[pred[img]['label']])

    return np.array(pred_labels), np.array(true_labels)
    

def accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    correct = len(np.where(pred == target)[0])
    acc = correct / len(pred)
    return acc
    
def get_confidence(pred: Dict, num_classes: int) -> List:
    count = np.array([0] * (num_classes + 1))
    confidence = np.array([0] * (num_classes + 1), dtype=float)
    for prediction in pred.values():
        if not isinstance(prediction, Dict):
            continue
        idx = prediction['label']
        conf = prediction['probs'][idx]
        confidence[idx] += conf
        count[idx] += 1
    confidence[num_classes] = sum(confidence)
    count[num_classes] = sum(count)
    confidence = confidence / count
    return confidence.tolist(), count.tolist()

if __name__ == '__main__':
    args = argument()
    suffix = 'dp' if args.dp else 'non_dp'
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    file_name = f'{args.result_dir}/zero-shot_{args.dataset}_{suffix}.json'
    images = make_images(args.data_dir, args.dataset)
    texts = get_texts(args.dataset)
    if not os.path.exists(file_name):
        predictions = predict(images, texts, args.device)
    else:
        with open(file_name, 'r') as f:
            predictions = json.load(f)

    pred, target = get_label(predictions, args.dataset)
    confidence, pred_dist = get_confidence(predictions, args.num_classes)
    predictions['accuracy'] = accuracy(pred, target)
    predictions['confidence'] = confidence
    predictions['predicted_distribution'] = pred_dist

    with open(file_name, 'w') as f:
        json.dump(predictions, f)
    


    