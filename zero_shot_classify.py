import os
import argparse
from typing import Dict, List
from transformers import CLIPProcessor, CLIPModel
from transformers import GPT2ForQuestionAnswering, GPT2Tokenizer
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
        '--modality',
        type=str,
        choices=['image', 'text'],
        required=True
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

def get_samples(dir: str, modality: str) -> List:
    """
    Get a list of target samples
    """
    EXTENSIONS={
    'image':
        ['jpg', 'jpeg', 'png', 'gif'],
    'text':
        ['txt']
    }
    data = []
    for (root, dirs, files) in os.walk(dir):
        for file in files:
            if file.split('.')[-1] not in EXTENSIONS[modality]:
                continue
            full = os.path.join(root, file)
            data.append(full)     
    return data

def get_texts(config: Dict, num: int = -1) -> List:
    """
    Get a list of text for classification
    """
    texts = []
    base = config['base']
    if num > 0:
        texts = [base for _ in range(num)]
    else:
        for label in config['labels']:
            text = base.replace('LABEL', label)
            texts.append(text)
    return texts

def predict(samples: List, texts: List[str], device_num: int, model_name: str, checkpoint: str) -> Dict:
    checkpoint = "openai/clip-vit-large-patch14"
    
    if model_name == 'clip':
        predictions = clip_predict(samples, texts, device_num, checkpoint)
    elif model_name == 'gpt':
        pass
    elif model_name == 'chatgpt':
        pass

    return predictions

def gpt_predict(samples: List[str], texts: List[str], device_num: int, checkpoint: str) -> Dict:
    device = f'cuda:{device_num}'
    processor = GPT2Tokenizer.from_pretrained(checkpoint)
    model = GPT2ForQuestionAnswering.from_pretrained(checkpoint)
    model.to(device)
    pass
    

def clip_predict(samples: List, texts: List[str], device_num: int, checkpoint: str) -> Dict:
    device = f'cuda:{device_num}'
    processor = CLIPProcessor.from_pretrained(checkpoint)
    model = CLIPModel.from_pretrained(checkpoint)
    model.to(device)

    predictions = {}
    for sample_path in samples:
        image = Image.open(sample_path)
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
        predictions[sample_path] = {
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
    images = get_samples(args.data_dir, args.dataset)
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
    


    