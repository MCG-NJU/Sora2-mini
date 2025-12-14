import os
import argparse
import random
import numpy as np
import sys
import json

import logging
import torch
from einops import rearrange
from omegaconf import OmegaConf
from safetensors.torch import safe_open
from torch.utils.data.distributed import DistributedSampler
from transformers import Qwen2_5OmniToken2WavBigVGANModel

from utils.eval_dataset import (
    EvalDataset, 
    collate_fn, 
    build_vocab_mapper, 
    list_str_to_idx
)
from utils.mel_dataset import get_mel_spectrogram
from utils.dis_util import get_world_size, get_local_rank, get_global_rank
from utils.io_utils import save_video
from wan import T5EncoderModel
from wan import UniAVGen as Model
from wan.configs import WAN_CONFIGS
from wan.modules.vae2_2 import Wan2_2_VAE
from wan.infer import UniAVGenPipeline


AVG = 0
RAVG = 1
A2V = 2
V2A = 3

def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)
   
def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--task", type=int, default=0, help="specific task")

    parser.add_argument("--config_file",
                        type=str,
                        default="configs/inference.yaml")
    
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    
    if getattr(args, "local_rank", -1) == -1:
        env_lr = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
        try:
            if env_lr is not None:
                args.local_rank = int(env_lr)
        except ValueError:
            pass
    
    if torch.cuda.is_available() and getattr(args, "local_rank", -1) >= 0:
        try:
            torch.cuda.set_device(args.local_rank % torch.cuda.device_count())
        except Exception:
            pass

    return args

@torch.no_grad()
def encode_images(vae, cond_video, ref_image, dtype):
    ref_latent = vae.encode([ref_image[0][:,:1]])[0].unsqueeze(0).to(dtype=dtype)
    if cond_video[0] is not None:
        cond_latent = vae.encode([cond_video[0]])[0].unsqueeze(0).to(dtype=dtype)
    else: 
        cond_latent = ref_latent
    return cond_latent, ref_latent

@torch.no_grad()
def run_sample(config, args, model_cfg, model, 
                   t5_model, t5_tokenizer, 
                   vae, bigvgan, vocab_mapper, 
                   eval_batch, device='cuda', dtype=torch.bfloat16
):
    data_id = eval_batch["data_id"][0]
    lang = eval_batch["lang"][0]

    video_prompt = eval_batch["text_prompt"]     

    cond_video = eval_batch["cond_video"]    
    if args.task == V2A and cond_video[0] is None: 
        raise ValueError("V2A needs condition video")  
    elif cond_video[0] is not None: 
        cond_video = [cv.to(device=device, dtype=dtype, non_blocking=True) for cv in cond_video]
       
    ref_image = eval_batch["ref_img"]
    if args.task != V2A and ref_image[0] is None: 
        raise ValueError("ref image is missed")  
    elif ref_image[0] is None: 
        ref_image = [cond_video[0][:, :1]]    
    ref_image = [ri.to(device=device, dtype=dtype, non_blocking=True) for ri in ref_image]

    text_char = eval_batch["text_char"]     
    ref_text_char = eval_batch["ref_text_char"]     
    cat_text_char = eval_batch["cat_text_char"]   
    neg_text_char = [[""]]

    cond_audio = eval_batch['cond_audio']    
    if args.task == A2V and cond_audio[0] is None: 
        raise ValueError("A2V needs condition audio")  
    elif cond_audio[0] is not None: 
        cond_audio = [ca.to(device=device, dtype=dtype, non_blocking=True) for ca in cond_audio]
    
    ref_audio = eval_batch["ref_audio"]
    if args.task == RAVG and ref_audio[0] is None: 
        raise ValueError("RAVG needs ref audio")     
    elif ref_audio[0] is not None: 
        ref_audio = [ra.to(device=device, dtype=dtype, non_blocking=True) for ra in ref_audio]

    sample_rate = 24000
    bsz = 1
    
    latent_cond_num = 1 if args.task != V2A else 21                           
    
    """ VAE """
    cond_video_embeds, ref_embedding = encode_images(vae, cond_video, ref_image, dtype)
    
    """ T5 & CLIP """
    text_ids, attention_mask = t5_tokenizer(video_prompt, return_mask=True, add_special_tokens=True)
    neg_text_ids, neg_attention_mask = t5_tokenizer([config.video_negative_prompt], return_mask=True, add_special_tokens=True)
    prompt_embeds = t5_model.text_embedding(text_ids, attention_mask, ref_embedding.device)[0]
    neg_prompt_embeds = t5_model.text_embedding(neg_text_ids, neg_attention_mask, ref_embedding.device)[0]
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=ref_embedding.device).unsqueeze(0)
    neg_prompt_embeds = neg_prompt_embeds.to(dtype=dtype, device=ref_embedding.device).unsqueeze(0)
                
    """ char to idx """
    ref_text_idx, _ = list_str_to_idx(ref_text_char, vocab_mapper)
    if args.task == RAVG:
        cat_text_idx, cat_text_lens = list_str_to_idx(cat_text_char, vocab_mapper) 
        neg_text_char = [ref_text_char[0]+(" " if lang == 'en' else [""])]
    else:
        cat_text_idx, cat_text_lens = list_str_to_idx(text_char, vocab_mapper) 

    if cat_text_idx.size(1) < 10:
        pad_idx = -1 * torch.ones(bsz, 10-cat_text_idx.size(1)).to(cat_text_idx)
        cat_text_idx = torch.cat([cat_text_idx, pad_idx], dim=1)
        cat_text_lens = [x.size(0) for x in cat_text_idx]
    ref_text_idx = ref_text_idx.to(device=device, dtype=torch.long)
    cat_text_idx = cat_text_idx.to(device=device, dtype=torch.long)
    cat_text_lens = torch.from_numpy(np.array(cat_text_lens)).long().to(device=device)

    neg_text_idx, neg_text_lens = list_str_to_idx(neg_text_char, vocab_mapper)
    if neg_text_idx.size(1) < 10:
        pad_idx = -1 * torch.ones(bsz, 10-neg_text_idx.size(1)).to(neg_text_idx)
        neg_text_idx = torch.cat([neg_text_idx, pad_idx], dim=1)
        neg_text_lens = [x.size(0) for x in neg_text_idx]
    neg_text_idx = neg_text_idx.to(device=device, dtype=torch.long)
    neg_text_lens = torch.from_numpy(np.array(neg_text_lens)).long().to(device=device)
    
    """ mel encoding """
    ref_audio_embeds = []
    if args.task == RAVG:
        for audio in ref_audio:
            audio = audio.to(dtype=dtype)
            ref_audio_embed = get_mel_spectrogram(audio.unsqueeze(0), 
                                        sampling_rate=sample_rate, 
                                        hop_size=sample_rate // 100).to(audio).detach() 
        ref_audio_embed = (ref_audio_embed - model_cfg.audio_mean) / model_cfg.audio_std
        ref_audio_embeds.append(rearrange(ref_audio_embed[0], "c t -> t c"))
    else: ref_audio_embeds = [None]

    cond_audio_embeds = []
    if args.task == A2V:
        for audio in cond_audio:
            audio = audio.to(dtype=dtype)
            cond_audio_embed = get_mel_spectrogram(audio.unsqueeze(0), 
                                        sampling_rate=sample_rate, 
                                        hop_size=sample_rate // 100).to(audio).detach() 
        cond_audio_embed = (cond_audio_embed - model_cfg.audio_mean) / model_cfg.audio_std
        cond_audio_embeds.append(rearrange(cond_audio_embed[0], "c t -> t c"))
    else: cond_audio_embeds = [None]
  
    audio_latent_cond_num = 0 if args.task != A2V else len(cond_audio_embeds[0])
    
    if args.task == RAVG:
        ref_audio_embeds = ref_audio_embeds[0].unsqueeze(0).to(device=device)
    num_tokens_video = 21
    height = ref_embedding.size(-2) * 16
    width = ref_embedding.size(-1) * 16
    num_tokens_audio =  506
    
    uniavgen_pipeline = UniAVGenPipeline(model, device=device, dtype=dtype)
    logging.info("Generating video ...")

    pred_embed, pred_mel = uniavgen_pipeline.generate(
        # ----- Video Generation -----
        text_embedding_video=prompt_embeds, 
        negative_text_embedding_video=neg_prompt_embeds,
        ref_image_embedding=ref_embedding, 
        cond_video_embedding=cond_video_embeds, 
        
        # ----- Audio Generation -----
        text_embedding=cat_text_idx,
        negative_text_embedding=neg_text_idx, 
        ref_audio_embedding=ref_audio_embeds, 
        cond_audio_embedding=cond_audio_embeds, 
        
        # ----- Settings -----
        num_tokens_audio=num_tokens_audio, 
        num_tokens_video=num_tokens_video, 
        height=height, width=width, 
        text_lens=cat_text_lens,
        neg_text_lens=neg_text_lens, 
        shift=config.shift,
        sampling_steps=config.num_steps,
        video_guide_scale=config.video_guidance_scale,        
        audio_guide_scale=config.audio_guidance_scale,     
        seed=config.seed,
        video_cond=latent_cond_num, 
        audio_cond=audio_latent_cond_num,
        macfg_prop=config.macfg_prop,
        slg=config.slg_layer,
        task=args.task, 
    )
    
    # ----- Decode video latent
    pred_embed[0][:, :latent_cond_num] = cond_video_embeds[0]
    pred_video = vae.decode([pred_embed[0]])
    pred_video = (pred_video[0] + 1.0) * 127.5      
        
    # ----- Decode audio mel
    if args.task == A2V:
        pred_mel[0][:audio_latent_cond_num] = cond_audio_embeds[0][:audio_latent_cond_num]
    pred_mel = pred_mel * model_cfg.audio_std + model_cfg.audio_mean
    pred_wav = bigvgan(pred_mel.permute(0, 2, 1).to(device=device, dtype=dtype))

    save_path = os.path.join(config.output_dir, '%s.mp4' % (data_id))
    save_video(save_path, pred_video.data.cpu().numpy().astype(np.uint8), pred_wav.reshape(-1).detach().float().cpu().numpy(), sample_rate=sample_rate, fps=16)


def main(args, config):

    #--------------- Dist.&Seed Init 
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    _init_logging(global_rank)

    torch.distributed.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=global_rank,
        world_size=world_size)

    args.local_rank = local_rank
    args.device = device

    #--------------- Data Construct 
    eval_dataset = EvalDataset( 
        meta_file=config.test_csv, 
        sampling_rate=24000, 
        max_volume=0.999146, 
    )
    
    sampler = DistributedSampler(eval_dataset)

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False, 
        collate_fn=collate_fn, 
        sampler=sampler
    )

    #----------------- Model Init&Load   
    weight_dtype = torch.bfloat16
    model_cfg = WAN_CONFIGS['ti2v-5B']

    checkpoint_dir = config.model_path
    text_encoder = T5EncoderModel(
        text_len=model_cfg.text_len,
        dtype=model_cfg.t5_dtype,
        device=args.device,
        checkpoint_path=os.path.join(checkpoint_dir, model_cfg.t5_checkpoint),
        tokenizer_path=os.path.join(checkpoint_dir, model_cfg.t5_tokenizer),
        shard_fn=None,
    )
    text_encoder.model = text_encoder.model.eval().to(args.device, dtype=weight_dtype)

    vae = Wan2_2_VAE(
        vae_pth=os.path.join(checkpoint_dir, model_cfg.vae_checkpoint),
        device=args.device
    )
    vae.model = vae.model.eval().to(args.device, dtype=weight_dtype)

    model = Model(
            model_type='ti2v',
            patch_size=model_cfg.patch_size,
            text_len=model_cfg.text_len,
            in_dim=model_cfg.in_dim,  # Use config value: 48 dims for Wan2.2
            in_audio_dim=80,
            audio_cond_dim=384*5, 
            dim=model_cfg.dim,
            ffn_dim=model_cfg.ffn_dim,
            freq_dim=model_cfg.freq_dim,
            text_dim=model_cfg.text_dim,  # Use config value: 4096
            out_dim=model_cfg.out_dim,  # Use config value: 48 dims for Wan2.2
            out_dim_audio=80,
            num_heads=model_cfg.num_heads,
            num_layers=model_cfg.num_layers,
            window_size=model_cfg.window_size,
            qk_norm=model_cfg.qk_norm,
            cross_attn_norm=model_cfg.cross_attn_norm,
            eps=model_cfg.eps
        )
    
    checkpoint_file = config.model_path
    index_file = os.path.join(checkpoint_file, 'diffusion_pytorch_model.safetensors.index.json') 

    model_tensor = {}

    with open(index_file, 'r') as f:
        index_data = json.load(f)

    shard_files = set(index_data['weight_map'].values())
    for shard_file in shard_files:
        shard_path = os.path.join(checkpoint_file, shard_file)
        with safe_open(shard_path, framework="pt") as f:
            for k in f.keys():
                model_tensor[k] = f.get_tensor(k)

    model.load_state_dict(model_tensor, strict=False)

    model.requires_grad_(False)
    model = model.eval().to(args.device, dtype=weight_dtype)

    del model_tensor

    code2wav_bigvgan_model = Qwen2_5OmniToken2WavBigVGANModel.from_pretrained(
        os.path.join(checkpoint_dir, model_cfg.bigvgan), attn_implementation="sdpa")
    code2wav_bigvgan_model = code2wav_bigvgan_model.eval().to(args.device, dtype=weight_dtype)

    vocab_mapper, _ = build_vocab_mapper()


    #------------------------ Sample 
    if args.local_rank <= 0:
        output_dir = config.get("output_dir", "./outputs")
        os.makedirs(output_dir, exist_ok=True)

    for batch in dataloader:
        run_sample(config, args, model_cfg, model, 
                        text_encoder, text_encoder.tokenizer, vae, 
                        code2wav_bigvgan_model, vocab_mapper, 
                        batch, args.device, weight_dtype)

        torch.distributed.barrier()


if __name__ == "__main__":
    args = parse_args()
    config = OmegaConf.load(args.config_file)
    main(args, config)
