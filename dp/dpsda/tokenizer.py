from typing import List
import numpy as np

def tokenize(model: str, text: str) -> np.ndarray:
    if model == 'clip_vit_b_32':
        import clip
        arr = clip.tokenize(text, truncate=True).numpy().squeeze()
    elif model == 'bert_base_nli_mean_tokens':
        pass
    else:
        raise Exception("Unknown model for tokenizer")

    return arr