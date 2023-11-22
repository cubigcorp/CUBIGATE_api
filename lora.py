from diffusers import StableDiffusionPipeline
from diffusers.loaders import AttnProcsLayers
import os
import torch
from LoraDataset import LoraDataset
from torch.utils.data import DataLoader
from diffusers.models.attention_processor import LoRAAttnProcessor
import torch.nn.functional as F
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import time
import argparse

def compute_snr(timesteps, scheduler):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr


def train_LoRA(model_name:str, dir: str, output: str, lr:float, batch_size:int, epochs: int, snr_gamma:float, weight_decay:float, device: int) -> None:
    
    flag = True if os.path.exists(model_name) else False
    pipe = StableDiffusionPipeline.from_pretrained(model_name, torch_dtype=torch.float32)
    if flag: 
        pipe.save_pretrained(model_name)
    
    scheduler = pipe.scheduler
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    unet = pipe.unet
    print(unet)

    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    unet.to(f'cuda:{device}')
    vae.to(f'cuda:{device}')
    text_encoder.to(f'cuda:{device}')
    
    
    #LoRA
    lora_attn_procs = {}

    for name in unet.attn_processors.keys():
            cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            lora_attn_procs[name] = LoRAAttnProcessor(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                rank=8,
            ).to(f"cuda:{device}")

    unet.set_attn_processor(lora_attn_procs)

    lora_layers = AttnProcsLayers(unet.attn_processors)

    optimizer = torch.optim.AdamW(lora_layers.parameters(), lr=lr, weight_decay=weight_decay)
    
    dataset = LoraDataset(dir, pipe.feature_extractor, pipe.tokenizer)
    
    pixel, ids = next(iter(dataset))
    print(pixel)
    print(ids)

    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=100)
    total_step = epochs * len(dataloader)
    lr_scheduler = get_scheduler(name='cosine', optimizer=optimizer, num_warmup_steps=500, num_training_steps=total_step)
    progress_bar = tqdm(range(total_step))
    progress_bar.set_description("Steps")


    
    unet.train()
    for epoch in range(epochs):
        
        train_loss = 0.0
        for step, batch in enumerate(dataloader):

            pixel, input_ids = batch
            input_ids = input_ids.to(f"cuda:{device}")
            pixel = pixel.to(f"cuda:{device}")
        
            #image to latent space
            latents = vae.encode(pixel).latent_dist.sample().to(f"cuda:{device}")
            latents = latents * vae.config.scaling_factor

            #noise to add to the latents
            noise = torch.randn_like(latents, device=latents.device)
            bsz = latents.shape[0]
            #random timestep for each image
            timesteps = torch.randint(low=0, high=scheduler.config.num_train_timesteps, size=(bsz,), device=latents.device)
            timesteps = timesteps.long()
            #add noise to the latents according to the noise magnitude at each timestep
            noisy_latent = scheduler.add_noise(original_samples=latents, noise=noise, timesteps=timesteps)
            #text embedding for conditioning
            encoder_hidden_states = text_encoder(input_ids)[0]

            #target for loss
            target = noise
            model_pred = unet(noisy_latent, timesteps, encoder_hidden_states).sample

            snr = compute_snr(timesteps=timesteps, scheduler=scheduler)
            mse_loss_weights = (
                torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            )
            loss = F.mse_loss(model_pred.float(), target.float(), reduction='none')
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()

            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            logs = {"step":step, "step_loss":loss.detach().item(), "lr":lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            progress_bar.update(1)
            
        print(epoch)
                

    unet = unet.to(torch.float32)
    unet.save_attn_procs(f'lora/{output}')

    

if __name__ == '__main__':
    lr = 1e-4
    batch_size = 100
    epoch = 100
    snr_gamma = 0.5
    weight_decay = 1e-2
    
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', required=False, default = 1e-4, type=float, help="learning rate")
    parser.add_argument('--batch', required=False, default=75, type=int, help='batch size')
    parser.add_argument('--epoch', required=False, default=100, type=int, help="epoch")
    parser.add_argument('--decay', required=False, default=1e-2, type=float)
    parser.add_argument('--gamma', required=False, default=0.5, type=float)
    parser.add_argument('--dir', required=True, type=str)
    parser.add_argument('--output', required=True, type=str)
    parser.add_argument('--device', required=True, type=int)

    args = parser.parse_args()
    
    model_name = "runwayml/stable-diffusion-v1-5"
    
    train_LoRA(model_name=model_name, dir=args.dir, output=args.output, lr=args.lr, batch_size=args.batch, epochs=args.epoch, snr_gamma=args.gamma, weight_decay=args.decay, device=args.device)
    






        
        
        
        
        
        


        
