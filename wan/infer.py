# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import random
import numpy as np
from tqdm import tqdm
from functools import partial
import torch.distributed as dist

AVG = 0
RAVG = 1
A2V = 2
V2A = 3

def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def timestep_transform(
        t,
        shift=5.0,
        num_timesteps=1000,
):
    t = t / num_timesteps
    # shift the timestep based on ratio
    new_t = shift * t / (1 + (shift - 1) * t)
    new_t = new_t * num_timesteps
    return new_t


class UniAVGenPipeline:
    def __init__(
            self,
            wan_model,
            device='cuda',
            dtype=torch.float16,
            num_timesteps=1000,
    ):
        self.model = wan_model
        self.param_dtype = dtype
        self.device = device
        self.num_timesteps = num_timesteps
        self.use_timestep_transform = True

    def add_noise(
            self,
            original_samples: torch.FloatTensor,
            noise: torch.FloatTensor,
            timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        """
        compatible with diffusers add_noise()
        """
        timesteps = timesteps.float() / self.num_timesteps
        timesteps = timesteps.view(timesteps.shape + (1,) * (len(noise.shape) - 1))

        return (1 - timesteps) * original_samples + timesteps * noise

    def generate(self,
                 # ----- Video Generation -----
                 text_embedding_video,              # For video gen
                 negative_text_embedding_video,      # For video gen
                 ref_image_embedding,               # For video gen
                 cond_video_embedding,              # For video gen
                 
                 # ----- Audio Generation -----
                 text_embedding,                    # For audio gen
                 negative_text_embedding,           # For audio gen
                 ref_audio_embedding,               # For audio gen
                 cond_audio_embedding,              # For audio gen
                 
                 # ----- Settings -----
                 num_tokens_audio,
                 num_tokens_video,
                 height, width, 
                 text_lens,
                 neg_text_lens,  
                 shift=5.0,
                 sampling_steps=50,
                 video_guide_scale=3.0,
                 audio_guide_scale=2.0,
                 seed=2025,
                 video_cond=0, 
                 audio_cond=0,
                 macfg_prop=1e4,
                 slg=-1,
                 task=AVG
                 ):
        
        seed = seed if seed >= 0 else random.randint(0, 99999999)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        
        # -----sample video noise
        batch_size = len(text_embedding_video)
        noise_video = torch.randn(
            batch_size, 48, num_tokens_video, height//16, width//16, 
            dtype=self.param_dtype,
            device=self.device)
        
        # ----- sample audio noise
        noise_audio = torch.randn(
            batch_size, num_tokens_audio, 80,
            dtype=self.param_dtype,
            device=self.device)

        # evaluation mode
        with torch.no_grad():
            timesteps = list(np.linspace(self.num_timesteps, 1, sampling_steps, dtype=np.float32))
            timesteps.append(0.)
            timesteps = [torch.tensor([t], device=self.device) for t in timesteps]
            if self.use_timestep_transform:
                timesteps = [timestep_transform(t, shift=shift, num_timesteps=self.num_timesteps).to(dtype=self.param_dtype) for t in
                             timesteps]
            # sample videos
            latent_audio = noise_audio
            latent_video = noise_video

            # prepare condition and uncondition configs
            arg_c = {
                'context_list': [
                    # ----- audio -----
                    text_embedding, 
                    ref_audio_embedding, 
                    cond_audio_embedding, 

                    # ----- video -----
                    ref_image_embedding, 
                    cond_video_embedding, 
                    text_embedding_video, 
                ],
                'seq_len': text_lens,
                'video_cond': video_cond,
                'audio_cond': audio_cond,
            }

            arg_null = {
                'context_list': [
                    # ----- audio -----
                    negative_text_embedding, 
                    ref_audio_embedding, 
                    cond_audio_embedding, 
 
                    # ----- video -----
                    ref_image_embedding, 
                    cond_video_embedding, 
                    negative_text_embedding_video,
                ],
                'seq_len': neg_text_lens,
                'video_cond': video_cond,
                'audio_cond': audio_cond,
            }

            if slg:
                arg_null['slg'] = slg
            
            if task == RAVG:
                arg_c['no_v2a'] = True 
            
            if task == A2V:
                arg_c['no_v2a'] = True 

            if task == V2A:
                arg_c['no_a2v'] = True

            progress_wrap = partial(tqdm, total=len(timesteps) - 1)
            for i in progress_wrap(range(len(timesteps) - 1)):
                if i < int(macfg_prop * sampling_steps):
                    arg_null['zero_con'] = True 
                else: 
                    arg_null['zero_con'] = False 
                
                if task == RAVG:
                    arg_c['no_v2a'] = True 

                if task == A2V:
                    arg_null['no_v2a'] = True 

                if task == V2A:
                    arg_null['no_a2v'] = True
                
                timestep = timesteps[i]
                latent_audio_model_input = latent_audio.to(self.device, dtype=self.param_dtype)
                latent_video_model_input = latent_video.to(self.device, dtype=self.param_dtype)

                
                noise_video_pred_cond, noise_audio_pred_cond = self.model(
                    latent_video_model_input, 
                    [latent_audio_model_input[0]], 
                    t=timestep, **arg_c
                )
                noise_video_pred_cond = noise_video_pred_cond[0].unsqueeze(0)
                noise_audio_pred_cond = noise_audio_pred_cond[0]
                torch_gc()
                
                
                noise_video_pred_uncond, noise_audio_pred_uncond = self.model(
                    latent_video_model_input, 
                    [latent_audio_model_input[0]], 
                    t=timestep, **arg_null
                )
                noise_video_pred_uncond = noise_video_pred_uncond[0].unsqueeze(0)
                noise_audio_pred_uncond = noise_audio_pred_uncond[0]
                torch_gc()
                
                # vanilla CFG strategy
                noise_audio_pred = noise_audio_pred_uncond + audio_guide_scale * (
                        noise_audio_pred_cond - noise_audio_pred_uncond)
                noise_video_pred = noise_video_pred_uncond + video_guide_scale * (
                        noise_video_pred_cond - noise_video_pred_uncond)

                noise_audio_pred = -noise_audio_pred
                noise_video_pred = -noise_video_pred
                
                # update latent
                dt = timesteps[i] - timesteps[i + 1]
                dt = dt / self.num_timesteps

                latent_audio = latent_audio + noise_audio_pred * dt[:, None, None]
                latent_video = latent_video + noise_video_pred[:, :, 1:] * dt[:, None, None]

                x0 = [latent_video.to(self.device), latent_audio.to(self.device)]
                del latent_audio_model_input, latent_video_model_input, timestep

            torch_gc()
        
        if dist.is_initialized():
            dist.barrier()

        del noise_audio, noise_video, latent_audio, latent_video
        torch_gc()

        return x0




