import numpy as np
from typing import List

def tokenize(model: str, text: str) -> np.ndarray:
    if model == 'clip_vit_b_32':
        import clip
        arr = clip.tokenize(text, truncate=True).numpy().squeeze()
    elif model == 'bert_base_nli_mean_tokens':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
        arr = tokenizer.encode_plus(text, padding="max_length", truncation=True, return_tensors='np', max_length=42)['input_ids'].squeeze()
    elif model == "all_mpnet_base_v2":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        arr = tokenizer(text, padding="max_length", truncation=True, return_tensors='np', max_length=77)['input_ids'].squeeze()
    else:
        raise Exception("Unknown model for tokenizer")

    return arr

def detokenize(model: str, tokens: np.ndarray) -> List[str]:
    if model == 'sentence-transformers/bert-base-nli-mean-tokens':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(model)
    elif model == "sentence-transformers/all-mpnet-base-v2":
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
    else:
        raise Exception("Unknown model for tokenizer")
    texts = [tokenizer.decode(token) for token in tokens]
    return texts