import os
import argparse
from typing import Dict, List
from transformers import CLIPProcessor, CLIPModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import pipeline
from PIL import Image
import torch
import numpy as np
import json
import openai

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help="Path of the base data directory."
    )
    parser.add_argument(
        '--suffix',
        type=str,
        required=False,
        help="Suffix for output"
    )
    parser.add_argument(
        '--modality',
        type=str,
        choices=['image', 'text'],
        required=True
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        choices=['gpt', 'clip', 'bert', 'chatgpt'],
        help="Name of the pretrained model to use."
    )
    parser.add_argument(
        '--api_key_path',
        type=str,
        required=False,
        help="API key for chatGPT"
    )
    parser.add_argument(
        '--few_shot',
        type=int,
        required=False,
        help="Number of exemplars for few-shot learning"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help="Name of the pretrained model to use."
    )
    parser.add_argument(
        '--device',
        type=int,
        required=False,
        help="Index of the GPU device to put everything on."
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        required=True,
        help="Number of classes."
    )
    parser.add_argument(
        '--label_text',
        action='store_true',
        help="Whether there are label specific texts"
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

def get_texts(config: Dict, label_text: bool) -> List:
    """
    Get a list of text for classification

    Parameters
    ----------
    config:
        text dictionary
    num_sample:
        needed when there is no label specific prompt
    """
    texts = []
    base = config['base']
    if label_text :
        for label in config['labels']:
            text = base.replace('LABEL', label)
            texts.append(text)
    else:
        texts = base
    return texts

def predict(samples: List, texts, labels: List[str], device_num: int, model_name: str, checkpoint: str, few_shot: int=0, api_key_path: str=None) -> Dict:
    # checkpoint = "openai/clip-vit-large-patch14"
    
    if model_name == 'clip':
        predictions = clip_predict(samples, labels, device_num, checkpoint)
    elif model_name == 'gpt':
        predictions = gpt_predict(samples, labels, device_num, checkpoint)
    elif model_name == 'bert':
        predictions = bert_predict(samples, labels, device_num, checkpoint)
    elif model_name == 'chatgpt':
        with open(api_key_path, 'r') as f:
            api_key = f.read()
        predictions = chatgpt_predict(samples, texts, labels, checkpoint, few_shot, api_key)

    return predictions

def chatgpt_predict(samples: List[str], text: str, labels: List[str], checkpoint: str, few_shot: int=0, key: str=None) -> Dict:
    openai.api_key = key
    rng = np.random.default_rng(seed=2023)
    every = [*range(len(samples))]
    predictions = {}
    for idx, sample_path in enumerate(samples):
        candidate = every.copy()
        candidate.remove(idx)
        exem_idx = rng.choice(candidate, few_shot)
        exemplars = np.array(samples)[exem_idx]
        exem_prompt = ""
        for exemplar in exemplars:
            with open(exemplar, 'r') as f:
                exem = f.read()
            label = exemplar.split('/')[-2]
            exem_prompt += f'{exem} : {label}\n'
        with open(sample_path, 'r') as f:
            sample = f.read()
        unique_label = list(set(labels))
        label_text = f"Do not use any words other than {', '.join(unique_label[:-1])} or {unique_label[-1]}"
        prompt = f"{text} Answer in one word. {label_text} \n\n{exem_prompt}{sample} : "
        message = [
                    {"role": "user", "content": prompt }
                ]
        response = openai.ChatCompletion.create(
                  model=checkpoint, 
                  messages=message,
                  request_timeout = 1000)
        label = response.choices[0].message.content
        predictions[sample_path] = {
            'label': label.lower().strip()
        }
    return predictions
    

def bert_predict(samples: List[str], labels: List[str], device_num: int, checkpoint: str) -> Dict:
    device = f"cuda:{device_num}"
    tokenizer = BertTokenizer.from_pretrained(checkpoint)
    model = BertForSequenceClassification.from_pretrained(checkpoint)
    classifier = pipeline('zero-shot-classification', model=model, tokenizer=tokenizer, device=device)
    labels = list(set(labels))

    predictions = {}
    for sample_path in samples:
        with open(sample_path, 'r') as f:
            text = f.read()
        output = classifier(text, labels)
        prob = output['scores']
        idx = max(range(len(prob)), key=lambda i: prob[i])
        predictions[sample_path]= {
            'probs': prob,
            'label': idx
        }
    return predictions

def gpt_predict(samples: List[str], labels: List[str], device_num: int, checkpoint: str) -> Dict:
    device = f'cuda:{device_num}'
    classifier = pipeline("zero-shot-classification", model=checkpoint, device=device)
    labels = list(set(labels))

    predictions = {}
    for sample_path in samples:
        with open(sample_path, 'r') as f:
            text = f.read()
        output = classifier(text, labels)
        prob = output['scores']
        idx = max(range(len(prob)), key=lambda i: prob[i])
        predictions[sample_path]= {
            'probs': prob,
            'label': idx
        }

    return predictions

    

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

def get_label(pred: Dict, labels: List, by_index: bool):
    pred_labels = []
    true_labels = []

    for sample in pred.keys():
        temp = sample.split('/')
        if len(temp) < 2:
            continue
        true_label = sample.split('/')[-2]
        true_labels.append(true_label)
        if by_index:
            pred_labels.append(labels[pred[sample]['label']])
        else:
            pred_labels.append(pred[sample]['label'])

    return np.array(pred_labels), np.array(true_labels)
    

def get_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
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
    return confidence.tolist()

def get_dist(pred: Dict, num_classes: int, labels: List) -> [List, int]:
    count = np.array([0] * (num_classes + 1))
    for prediction in pred.values():
        if not isinstance(prediction, Dict):
            continue
        label = prediction['label']
        if label in labels:
            idx = labels.index(label)
        else:
            idx = num_classes
        count[idx] += 1
    count[num_classes] = sum(count)
    return count.tolist()

if __name__ == '__main__':
    args = argument()
    dp = 'dp' if args.dp else 'non_dp'
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)
    file_name = f'{args.result_dir}/zero-shot_{args.suffix}_{dp}.json'
    samples = get_samples(args.data_dir, args.modality)
    with open(os.path.join(args.data_dir, 'config')) as f:
        config = json.load(f)
    texts = get_texts(config['texts'], args.label_text)
    if not os.path.exists(file_name):
        predictions = predict(samples, texts, config['total_labels'], args.device, args.model_name, args.checkpoint, args.few_shot, args.api_key_path)
    else:
        with open(file_name, 'r') as f:
            predictions = json.load(f)
    by_index = False if args.model_name == 'chatgpt' else True  # ChatGPT는 라벨 이름을 그대로 받고 다른 모델은 인덱스로 받음
    pred, target = get_label(predictions, config['total_labels'], by_index)
    predictions['accuracy'] = get_accuracy(pred, target)
    if args.model_name != 'chatgpt':  
        confidence, pred_dist = get_confidence(predictions, args.num_classes)
        predictions['confidence'] = confidence
    else:
        pred_dist = get_dist(predictions, args.num_classes, config['total_labels'])
    predictions['predicted_distribution'] = pred_dist

    with open(file_name, 'w') as f:
        json.dump(predictions, f)
    


    