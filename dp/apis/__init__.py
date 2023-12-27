from .api import API


def get_api_class_from_name(name):
    # Lazy import to improve loading speed and reduce libary dependency.
    if name == 'DALLE':
        from .dalle_api import DALLEAPI
        return DALLEAPI
    elif name == 'stable_diffusion':
        from .stable_diffusion_api import StableDiffusionAPI
        return StableDiffusionAPI
    elif name == 'improved_diffusion':
        from .improved_diffusion_api import ImprovedDiffusionAPI
        return ImprovedDiffusionAPI
    elif name == 'gpt2':
        from .gpt2_api import GPT2API
        return GPT2API
    elif name == 'chatgpt':
        from .chat_gpt_api import ChatGPTAPI
        return ChatGPTAPI
    elif name == 'chat_llama2':
        from .chat_llama_api import ChatLlama2API
        return ChatLlama2API
    else:
        raise ValueError(f'Unknown API name {name}')


__all__ = ['get_api_class_from_name', 'API']
