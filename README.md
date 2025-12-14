<p align="center">

  <h2 align="center">UniAVGen: Unified Audio and Video Generation with <br> Asymmetric Cross-Modal Interactions</h2>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=48vfuRAAAAAJ&hl=zh-CN"><strong>Guozhen Zhang</strong></a>
    ·
    <a href="https://scholar.google.cz/citations?user=F2cnLlIAAAAJ&hl=zh-CN&oi=ao"><strong>Zixiang Zhou</strong></a>
    ·
    <a href="https://scholar.google.cz/citations?user=Jm5qsAYAAAAJ&hl=zh-CN&authuser=1"><strong>Teng Hu</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=gYTyZGYAAAAJ&hl=zh-CN&oi=sra"><strong>Ziqiao Peng</strong></a>
    ·
    <a href="https://github.com/angzong"><strong>Youliang Zhang</strong></a>
    <br>
    <a href="https://scholar.google.com/citations?user=dmdhJjgAAAAJ&hl=zh-CN"><strong>Yi Chen</strong></a>
    ·
    <a href="https://openreview.net/profile?id=~Yuan_Zhou12"><strong>Yuan Zhou</strong></a>
    ·
    <a href="https://openreview.net/profile?id=~Qinglin_Lu2"><strong>Qinglin Lu</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=HEuN8PcAAAAJ&hl=en"><strong>Limin Wang</strong></a>
    <br>
    <b></a>MCG-NJU &nbsp; | &nbsp; </a> Tencent Hunyuan  </b>
    <br><br>
        <a href="https://arxiv.org/pdf/2511.03334"><img src='https://img.shields.io/badge/arXiv-2511.03334-red' alt='Paper PDF'></a>
        <a href='https://mcg-nju.github.io/UniAVGen/'><img src='https://img.shields.io/badge/Project-Page-blue' alt='Project Page'></a>
        <a href='https://huggingface.co/MCG-NJU/UniAVGen'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow'></a>
    <br>
  </p>
</p>

This repository is the official implementation of paper "UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions". UniAVGen is a unified framework for high-fidelity joint audio-video generation, addressing key limitations of existing methods such as poor lip synchronization, insufficient semantic consistency, and limited task generalization.

At its core, UniAVGen adopts a symmetric dual-branch architecture (parallel Diffusion Transformers for audio and video) and introduces three critical innovations: (1) Asymmetric Cross-Modal Interaction for bidirectional temporal alignment, (2) Face-Aware Modulation to prioritize salient facial regions during interaction, (3) Modality-Aware Classifier-Free Guidance to amplify cross-modal correlations during inference.

![teaser](figs/teaser.png?raw=true)

## :boom: News


- **2025-12-14**: Released the inference code and [weights](https://huggingface.co/MCG-NJU/UniAVGen) of UniAVGen.
- **2025-11-05**: Our paper is in public on [arxiv](https://arxiv.org/pdf/2511.03334).


## 💕 Installation
```
git clone https://github.com/MCG-NJU/UniAVGen.git
cd UniAVGen

conda create -n uniavgen python=3.10 -y
conda activate uniavgen

# CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl && python -c "import flash_attn"
pip install "xfuser[diffusers,flash-attn]"
pip install -r requirements.txt
```

### Download Checkpoints
```
huggingface-cli download MCG-NJU/UniAVGen --local-dir ./UniAVGen
```

## 😎 Inference 
We support joint audio-visual generation (AVG, id: 0), joint generation with reference audio (RAVG, id: 1), audio-driven video generation (A2V, id: 2), and video-driven audio generation (V2A, id: 3).
### Inference Data Construct
Given the need to consider compatibility across multiple tasks, we organize data into CSV format, which supports both single-GPU and multi-GPU parallel testing. The definitions of each column in the CSV are provided below:
```yaml
data_id # [required] The name of output samples.
ref_image_path # [required] reference image is required for all tasks except V2A
speech_content # [required] audio speech content
prompt # [required] video caption
lang # [required] language, en or zh (performance of zh is under improvement)
ref_audio_path # [optional] reference audio is required only for RAVG
ref_speech_content # [optional] The speech content correspond to ref_audio
video_path # [optional] The condition video for V2A
audio_path # [optional] The condition audio for A2V
```
We provide demo CSVs for each task in the `examples/csvs`.

### Inference Config

You can modify `configs/inference.yaml` to control the parameters of the sampling process; the details are as follows:

```yaml
model_path: UniAVGen
audio_guidance_scale: 2.0   
video_guidance_scale: 3.0   
output_dir: ./outputs/demo
num_steps: 50
shift: 5.0
seed: 2025
video_negative_prompt: ""  
test_csv: examples/csvs/test_task_AVG.csv # path of test csv
slg_layer: 11        # skip layer guidance, default = 11
macfg_prop: 0.5       # proportion of timesteps using MA-CFG, default = 0.5
```
### Sample

```yaml
# Sigle GPU
torchrun --nnodes 1 --nproc_per_node 1 inference.py --task 0 # specify the task id
# Multi GPU
torchrun --nnodes 1 --nproc_per_node 8 inference.py --task 0 
```

## :muscle:	Citation

If you think this project is helpful in your research or for application, please feel free to leave a star⭐️ and cite our paper:

```BibTeX
@misc{zhang2025uniavgenunifiedaudiovideo,
      title={UniAVGen: Unified Audio and Video Generation with Asymmetric Cross-Modal Interactions}, 
      author={Guozhen Zhang and Zixiang Zhou and Teng Hu and Ziqiao Peng and Youliang Zhang and Yi Chen and Yuan Zhou and Qinglin Lu and Limin Wang},
      year={2025},
      eprint={2511.03334},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2511.03334}, 
}
```

## :heartpulse:	License and Acknowledgement

This project is released under the Apache 2.0 license. The codes are based on [Wan2.2](https://github.com/Wan-Video/Wan2.2), [F5TTS](https://github.com/SWivid/F5-TTS) and [OVI](https://github.com/character-ai/Ovi). Please also follow their licenses. Thanks for their awesome works.
